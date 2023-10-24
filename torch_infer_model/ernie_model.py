# -*- coding: utf-8 -*

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
# from ernie_utils import ErnieConfig
ACT_DICT = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
}

class ErnieModel(nn.Module):
    """ ernie model """

    def __init__(self, cfg):
        """
        Fundamental pretrained Ernie model
        """
        nn.Module.__init__(self)
        self.cfg = cfg
        d_model = cfg['hidden_size']
        d_emb = cfg.get('emb_size', cfg['hidden_size'])
        d_vocab = cfg['vocab_size']
        d_pos = cfg['max_position_embeddings']
        if cfg.has('sent_type_vocab_size'):
            d_sent = cfg['sent_type_vocab_size']
        else:
            d_sent = cfg.get('type_vocab_size', 2)

        self.n_head = cfg['num_attention_heads']
        self.return_additional_info = cfg.get('return_additional_info', False)

        self.word_emb = nn.Embedding(d_vocab, d_emb)
        self.pos_emb = nn.Embedding(d_pos, d_emb)

        self._use_sent_id = cfg.get('use_sent_id', True)
        if self._use_sent_id:
            self.sent_emb = nn.Embedding(d_sent, d_emb)

        self._use_task_id = cfg.get('use_task_id', False)
        if self._use_task_id:
            self._task_types = cfg.get('task_type_vocab_size', 3)
            logging.info('using task_id, #task_types:{}'.format(self._task_types))
            self.task_emb = nn.Embedding(self._task_types, d_emb)

        self.ln = nn.LayerNorm(d_model)

        prob = cfg['hidden_dropout_prob']
        self.dropout = nn.Dropout(p=prob)

        self.encoder_stack = ErnieEncoderStack(cfg)

        if cfg.get('has_pooler', True):
            self.pooler = nn.Linear(cfg['hidden_size'], cfg['hidden_size'])
        else:
            self.pooler = None

        self.register_buffer("position_ids", torch.arange(d_pos).expand((1, -1)), persistent=False)
        self.register_buffer("token_type_ids", torch.zeros_like(self.position_ids), persistent=False)

        #self.apply(self._init_weights)

    def forward(self,
                src_ids,
                sent_ids=None,
                pos_ids=None,
                input_mask=None,
                task_ids=None,
                attn_bias=None,
                past_cache=None,
                use_causal_mask=False):
        """
        Args:
            src_ids (`Variable` of shape `[batch_size, seq_len]`):
                Indices of input sequence tokens in the vocabulary.
            sent_ids (optional, `Variable` of shape `[batch_size, seq_len]`):
                aka token_type_ids, Segment token indices to indicate first and second portions of the inputs.
                if None, assume all tokens come from `segment_a`
            pos_ids(optional, `Variable` of shape `[batch_size, seq_len]`):
                Indices of positions of each input sequence tokens in the position embeddings.
            input_mask(optional `Variable` of shape `[batch_size, seq_len]`):
                Mask to avoid performing attention on the padding token indices of the encoder input.
            task_ids(optional `Variable` of shape `[batch_size, seq_len]`):
                task type for pre_train task type
            attn_bias(optional, `Variable` of shape `[batch_size, seq_len, seq_len] or False`):
                3D version of `input_mask`, if set, overrides `input_mask`; if set not False, will not apply attention mask
            past_cache(optional, tuple of two lists: cached key and cached value,
                each is a list of `Variable`s of shape `[batch_size, seq_len, hidden_size]`):
                cached key/value tensor that will be concated to generated key/value when performing self attention.
                if set, `attn_bias` should not be None.

        Returns:
            pooled (`Variable` of shape `[batch_size, hidden_size]`):
                output logits of pooler classifier
            encoded(`Variable` of shape `[batch_size, seq_len, hidden_size]`):
                output logits of transformer stack
            info (Dictionary):
                addtional middle level info, inclues: all hidden stats, k/v caches.
        """
        assert len(src_ids.shape) == 2, 'expect src_ids.shape = [batch, sequence], got %s' % (repr(src_ids.shape))
        assert attn_bias is not None if past_cache else True, 'if `past_cache` specified; attn_bias must not be None'
        device = src_ids.device
        batch_size, seq_length = src_ids.shape
        if pos_ids is None:
            pos_ids = self.position_ids[:, :seq_length]

        if attn_bias is None:
            if input_mask is None:
                input_mask = torch.ones((batch_size, seq_length), device=device)
            assert len(input_mask.shape) == 2
            input_mask = input_mask.unsqueeze(-1).to(torch.float32)
            attn_bias = input_mask.matmul(input_mask.transpose(-2, -1))
            if use_causal_mask:
                sequence = (torch.arange(0, seq_length, 1, dtype=torch.float32) + 1.).reshape([1, 1, -1, 1])
                causal_mask = (sequence.matmul(1. / sequence.transpose(-2, -1)) >= 1.).float()
                attn_bias *= causal_mask
        else:
            assert len(attn_bias.shape) == 3, 'expect attn_bias tobe rank 3, got %r' % attn_bias.shape

        attn_bias = (1. - attn_bias) * -10000.0
        attn_bias = attn_bias.unsqueeze(1).tile([1, self.n_head, 1, 1])  # avoid broadcast =_=

        if sent_ids is None:
            sent_ids = self.token_type_ids[:, :seq_length].expand(batch_size, seq_length)

        src_embedded = self.word_emb(src_ids)
        pos_embedded = self.pos_emb(pos_ids)
        embedded = src_embedded + pos_embedded
        if self._use_sent_id:
            sent_embedded = self.sent_emb(sent_ids)
            embedded = embedded + sent_embedded
        if self._use_task_id:
            task_embedded = self.task_emb(task_ids)
            embedded = embedded + task_embedded

        embedded = self.ln(embedded)
        embedded = self.dropout(embedded)

        encoded, hidden_list, cache_list = self.encoder_stack(embedded, attn_bias, past_cache=past_cache)

        if self.pooler is not None:
            pooled = torch.tanh(self.pooler(encoded[:, 0, :]))
        else:
            pooled = None

        additional_info = {
            'hiddens': hidden_list,
            'caches': cache_list,
        }

        if self.return_additional_info:
            return pooled, encoded, additional_info
        return pooled, encoded

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=self.cfg['initializer_range'])
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=self.cfg['initializer_range'])
            # if module.padding_idx is not None:
            #     module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ErnieEncoderStack(nn.Module):
    """ ernie encoder stack """

    def __init__(self, cfg):
        super(ErnieEncoderStack, self).__init__()
        n_layers = cfg['num_hidden_layers']
        self.block = nn.ModuleList([
            ErnieBlock(cfg)
            for i in range(n_layers)
        ])

    def forward(self, inputs, attn_bias=None, past_cache=None):
        """ forward function """
        if past_cache is not None:
            assert isinstance(
                past_cache, tuple
            ), 'unknown type of `past_cache`, expect tuple or list. got %s' % repr(type(past_cache))
            past_cache = list(zip(*past_cache))
        else:
            past_cache = [None] * len(self.block)
        cache_list_k, cache_list_v, hidden_list = [], [], [inputs]

        for b, p in zip(self.block, past_cache):
            inputs, cache = b(inputs, attn_bias=attn_bias, past_cache=p)
            cache_k, cache_v = cache
            cache_list_k.append(cache_k)
            cache_list_v.append(cache_v)
            hidden_list.append(inputs)

        return inputs, hidden_list, (cache_list_k, cache_list_v)



class ErnieBlock(nn.Module):
    """ ernie block class """

    def __init__(self, cfg):
        super(ErnieBlock, self).__init__()
        d_model = cfg['hidden_size']
        self.attn = AttentionLayer(cfg)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForwardLayer(cfg)
        self.ln2 = nn.LayerNorm(d_model)
        prob = cfg.get('intermediate_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = nn.Dropout(p=prob)

    def forward(self, inputs, attn_bias=None, past_cache=None):
        """ forward """
        attn_out, cache = self.attn(inputs, inputs, inputs, attn_bias, past_cache=past_cache)  # self attention
        attn_out = self.dropout(attn_out)
        hidden = attn_out + inputs
        hidden = self.ln1(hidden)  # dropout/ add/ norm

        ffn_out = self.ffn(hidden)
        ffn_out = self.dropout(ffn_out)
        hidden = ffn_out + hidden
        hidden = self.ln2(hidden)
        return hidden, cache


class AttentionLayer(nn.Module):
    """ attention layer """

    def __init__(self, cfg):
        super(AttentionLayer, self).__init__()
        # initializer = nn.initializer.TruncatedNormal(std=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        n_head = cfg['num_attention_heads']
        assert d_model % n_head == 0
        d_model_q = cfg.get('query_hidden_size_per_head', d_model // n_head) * n_head
        d_model_v = cfg.get('value_hidden_size_per_head', d_model // n_head) * n_head

        self.n_head = n_head
        self.d_key = d_model_q // n_head

        self.q = nn.Linear(d_model, d_model_q)
        self.k = nn.Linear(d_model, d_model_q)
        self.v = nn.Linear(d_model, d_model_v)
        self.o = nn.Linear(d_model_v, d_model)
        self.dropout = nn.Dropout(p=cfg['attention_probs_dropout_prob'])

    def forward(self, queries, keys, values, attn_bias, past_cache):
        """ layer forward function """
        assert len(queries.shape) == len(keys.shape) == len(values.shape) == 3
        # bsz, q_len, q_dim = queries.shape
        # bsz, k_len, k_dim = keys.shape
        # bsz, v_len, v_dim = values.shape
        # assert k_len == v_len

        q = self.q(queries)
        k = self.k(keys)
        v = self.v(values)

        cache = (k, v)
        if past_cache is not None:
            cached_k, cached_v = past_cache
            k = torch.concat([cached_k, k], 1)
            v = torch.concat([cached_v, v], 1)

        # [batch, head, seq, dim]
        q = q.reshape([q.shape[0], q.shape[1], self.n_head, q.shape[-1] // self.n_head]).permute([0, 2, 1, 3])
        # [batch, head, seq, dim]
        k = k.reshape([k.shape[0], k.shape[1], self.n_head, k.shape[-1] // self.n_head]).permute([0, 2, 1, 3])
        # [batch, head, seq, dim]
        v = v.reshape([v.shape[0], v.shape[1], self.n_head, v.shape[-1] // self.n_head]).permute([0, 2, 1, 3])
        # q = q.scale(self.d_key ** -0.5)
        score = q.matmul(k.transpose(-2, -1))
        score = score / (self.d_key ** 0.5)

        if attn_bias is not None:
            score += attn_bias
        score = F.softmax(score, dim=-1)
        score = self.dropout(score)

        out = score.matmul(v).permute([0, 2, 1, 3])
        out = out.reshape([out.shape[0], out.shape[1], out.shape[2] * out.shape[3]])
        out = self.o(out)
        return out, cache

class PositionWiseFeedForwardLayer(nn.Module):
    """ post wise feed forward layer """

    def __init__(self, cfg):
        super(PositionWiseFeedForwardLayer, self).__init__()
        d_model = cfg['hidden_size']
        d_ffn = cfg.get('intermediate_size', 4 * d_model)

        self.act = ACT_DICT[cfg['hidden_act']]()
        self.i = nn.Linear(d_model, d_ffn)
        self.o = nn.Linear(d_ffn, d_model)
        prob = cfg.get('intermediate_dropout_prob', 0.)
        self.dropout = nn.Dropout(p=prob)

    def forward(self, inputs):
        """ forward """
        hidden = self.act(self.i(inputs))
        hidden = self.dropout(hidden)
        out = self.o(hidden)
        return out

class Aside(nn.Module):
    def __init__(self, cfg):
        super(Aside, self).__init__()
        self.embed0 = nn.Embedding(cfg['aside_emb_0_i'], cfg['aside_emb_0_o']) ## 1432 20
        self.embed1 = nn.Embedding(cfg['aside_emb_1_i'], cfg['aside_emb_1_o'])# 11 20
        self.embed2 = nn.Embedding(cfg['aside_emb_2_i'], cfg['aside_emb_2_o'])# 11 20
        self.embed3 = nn.Embedding(cfg['aside_emb_3_i'], cfg['aside_emb_3_o'])# 13 20
        self.embed4 = nn.Embedding(cfg['aside_emb_4_i'], cfg['aside_emb_4_o'])# 11 20
        self.embed5 = nn.Embedding(cfg['aside_emb_5_i'], cfg['aside_emb_5_o'])# 11 20
        self.embed6 = nn.Embedding(cfg['aside_emb_6_i'], cfg['aside_emb_6_o'])# 11 20
        self.embed7 = nn.Embedding(cfg['aside_emb_7_i'], cfg['aside_emb_7_o'])# 11 20
        self.feature_emb_fc1 = nn.Linear(cfg['aside_embed_fc'], cfg['hidden_size'])# 160 768 .t()
        self.activation = nn.ReLU()

        self.feature_emb_fc2 = nn.Linear(cfg['hidden_size'], cfg['aside_cls_out'])# 768 384 .t()
        self.cls_out = nn.Linear(cfg['aside_cls_out'], 1)# 384 1

    def forward(self, vector_list):
        x0 = self.embed0(vector_list[0])
        x1 = self.embed1(vector_list[1])
        x2 = self.embed2(vector_list[2])
        x3 = self.embed3(vector_list[3])
        x4 = self.embed4(vector_list[4])
        x5 = self.embed5(vector_list[5])
        x6 = self.embed0(vector_list[6])
        x7 = self.embed7(vector_list[7])
        x_cat = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7), dim = 1)
        x = x_cat.reshape(-1, 1, 1, 160)

        x = self.feature_emb_fc1(x)
        x = self.activation(x)

        x = self.feature_emb_fc2(x)
        x = self.activation(x)

        x = self.cls_out(x)
        return x

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()

        self.aside = Aside(cfg)
        self.ernie_model = ErnieModel(cfg)

        self.cls_out = nn.Linear(cfg['hidden_size'], 1)# 768 1
        # self.cls_out.weight.data = WeightDict["cls_out_w"].t()
        # self.cls_out.bias.data = WeightDict["cls_out_b"]

    def forward(self, src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor, aside_tensor_list):
        import pdb
        pdb.set_trace()
        cls_aside_out = self.aside(aside_tensor_list)

        pooled, encoded = self.ernie_model(src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor)

        cls_out = self.cls_out(pooled)
        x = cls_out + cls_aside_out
        x = torch.sigmoid(x)
        return x
