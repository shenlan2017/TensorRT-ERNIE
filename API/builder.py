import argparse
import ctypes
import json
import numpy as np
import os
import os.path
import re
import sys
import time

import paddle

paddle.enable_static()

# TensorRT
import tensorrt as trt
#from calibrator import ErnieCalibrator as ErnieCalibrator
from trt_helper import *


"""
TensorRT Initialization
"""
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
# if not handle:
    # raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.so` on your LD_LIBRARY_PATH?")

slice_output_shape = None

trt.init_libnvinfer_plugins(TRT_LOGGER, "")
plg_registry = trt.get_plugin_registry()

def set_tensor_name(tensor, prefix, name):
    tensor.name = prefix + name

def set_output_name(layer, prefix, name, out_idx = 0):
    set_tensor_name(layer.get_output(out_idx), prefix, name)

def set_output_range(layer, maxval, out_idx = 0):
    layer.get_output(out_idx).set_dynamic_range(-maxval, maxval)

def get_mha_dtype(config):
    dtype = trt.float32
    if config.fp16:
        dtype = trt.float16
    # Multi-head attention doesn't use INT8 inputs and output by default unless it is specified.
    if config.int8 and config.use_int8_multihead and not config.is_calib_mode:
        dtype = trt.int8
    return int(dtype)

def build_attention_layer(network_helper, prefix, config, weights_dict, x, mask):

    local_prefix = prefix + "multi_head_att_"

    # num_heads = config.num_heads
    # head_size = config.hidden_size // num_heads
    num_heads = 12
    head_size = 64  # 768 / 12

    # network_helper.markOutput(x)
    q_w = weights_dict[local_prefix + "query_fc.w_0"]
    q_b = weights_dict[local_prefix + "query_fc.b_0"]
    q = network_helper.addLinear(x, q_w, q_b)
    # network_helper.markOutput(q)
    q = network_helper.addShuffle(q, None, (0, -1, num_heads, head_size), (0, 2, 1, 3), "att_q_view_transpose")

    k_w = weights_dict[local_prefix + "key_fc.w_0"]
    k_b = weights_dict[local_prefix + "key_fc.b_0"]
    k = network_helper.addLinear(x, k_w, k_b)
    # network_helper.markOutput(k)
    k = network_helper.addShuffle(k, None, (0, -1, num_heads, head_size), (0, 2, 3, 1), "att_k_view_and transpose")
    # k = network_helper.addShuffle(k, None, (0, -1, self.h, self.d_k), (0, 2, 3, 1), "att_k_view_and transpose")

    v_w = weights_dict[local_prefix + "value_fc.w_0"]
    v_b = weights_dict[local_prefix + "value_fc.b_0"]
    v = network_helper.addLinear(x, v_w, v_b)
    # network_helper.markOutput(v)
    v = network_helper.addShuffle(v, None, (0, -1, num_heads, head_size), (0, 2, 1, 3), "att_v_view_and transpose")

    scores = network_helper.addMatMul(q, k, "q_mul_k")

    scores = network_helper.addScale(scores, 1/math.sqrt(head_size))

    scores = network_helper.addAdd(scores, mask)

    attn = network_helper.addSoftmax(scores, dim=-1)
    # attn = network_helper.addMaskedSoftmax(scores, mask, 1/math.sqrt(head_size), dim=-1)

    attn = network_helper.addMatMul(attn, v, "matmul(p_attn, value)")

    attn = network_helper.addShuffle(attn, (0, 2, 1, 3), (0, -1, 1, num_heads * head_size), None, "attn_transpose_and_reshape")

    # out_w = weights_dict[local_prefix + "out.weight"]
    # out_b = weights_dict[local_prefix + "out.bias"]
    out_w = weights_dict[local_prefix + "output_fc.w_0"]
    out_b = weights_dict[local_prefix + "output_fc.b_0"]

    attn_output = network_helper.addLinear(attn, out_w, out_b)

    return attn_output

def build_mlp_layer(network_helper, prefix, config, weights_dict, x):

    local_prefix = prefix + "ffn_"
    fc1_w = weights_dict[local_prefix + "fc_0.w_0"]
    fc1_b = weights_dict[local_prefix + "fc_0.b_0"]
    x = network_helper.addLinear(x, fc1_w, fc1_b)

    x = network_helper.addReLU(x)

    fc2_w = weights_dict[local_prefix + "fc_1.w_0"]
    fc2_b = weights_dict[local_prefix + "fc_1.b_0"]
    x = network_helper.addLinear(x, fc2_w, fc2_b)

    return x

def build_embeddings_layer(network_helper, weights_dict, src_ids_tensor, sent_ids_tensor, pos_ids_tensor):

    #  weight info
    #  transformer.embeddings.position_embeddings [1, 197, 768]
    #  transformer.embeddings.cls_token [1, 1, 768]
    #  transformer.embeddings.patch_embeddings.weight [768, 3, 16, 16]
    #  transformer.embeddings.patch_embeddings.bias [768]

    word_embedding = weights_dict["word_embedding"]
    sent_embedding = weights_dict["sent_embedding"]
    pos_embedding = weights_dict["pos_embedding"]

    src_embedded = network_helper.addEmbedding(src_ids_tensor, word_embedding, "word_embedding")
    pos_embedded = network_helper.addEmbedding(pos_ids_tensor, pos_embedding, "pos_embedding")
    sent_embedded = network_helper.addEmbedding(sent_ids_tensor, sent_embedding, "sent_embedding")

    x = network_helper.addAdd(src_embedded, pos_embedded)
    x = network_helper.addAdd(x, sent_embedded)

    return x


def build_block_layer(network_helper, prefix, config, weights_dict, x, mask):
    local_prefix = prefix
    h = x

    # self.attn
    x = build_attention_layer(network_helper, local_prefix, config, weights_dict, x, mask)

    x = network_helper.addAdd(x, h)

    # post_att_norm
    post_att_norm_weight = weights_dict[local_prefix + "post_att_layer_norm_scale"]
    post_att_norm_bias = weights_dict[local_prefix + "post_att_layer_norm_bias"]
    x = network_helper.addLayerNorm(x, post_att_norm_weight, post_att_norm_bias)

    h = x

    # fnn
    x = build_mlp_layer(network_helper, local_prefix, config, weights_dict, x)

    x = network_helper.addAdd(x, h)

    # post ffn_norm
    fnn_norm_weight = weights_dict[local_prefix + "post_ffn_layer_norm_scale"]
    fnn_norm_bias = weights_dict[local_prefix + "post_ffn_layer_norm_bias"]
    x = network_helper.addLayerNorm(x, fnn_norm_weight, fnn_norm_bias)

    return x

def build_encoder_layer(network_helper, prefix, config, weights_dict, x, mask):
    # for layer in range(0, config.encoder_num_layers):
    for layer in range(0, 12):
        local_prefix = prefix + "layer_{}_".format(layer)
        x = build_block_layer(network_helper, local_prefix, config, weights_dict, x, mask)
        # break

    x_shape_len = len(x.shape)
    start = np.zeros(x_shape_len, dtype=np.int32)
    #  start_weight = trt.Weights(start)
    start_tensor = network_helper.addConstant(start)

    slice_layer = network_helper.network.add_slice(x, start, start, (1, 1, 1, 1))
    slice_layer.set_input(1, start_tensor)
    slice_layer.set_input(2, slice_output_shape)
    sliced = slice_layer.get_output(0)
    print("sliced")

    pooled_w = weights_dict["pooled_fc.w_0"]
    pooled_b = weights_dict["pooled_fc.b_0"]
    x = network_helper.addLinear(sliced, pooled_w, pooled_b)

    return x

def build_ernie_model(network_helper, config, weights_dict, src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor):
    #  def forward(self, input_ids):
        #  embedding_output = self.embeddings(input_ids)
        #  encoded, attn_weights = self.encoder(embedding_output)
        #  return encoded, attn_weights
    prefix = "encoder_"
    embeddings = build_embeddings_layer(network_helper, weights_dict, src_ids_tensor, sent_ids_tensor, pos_ids_tensor)

    pre_encoder_norm_weight = weights_dict["pre_encoder_layer_norm_scale"]
    pre_encoder_norm_bias = weights_dict["pre_encoder_layer_norm_bias"]
    x = network_helper.addLayerNorm(embeddings, pre_encoder_norm_weight, pre_encoder_norm_bias)

    encoder_out = build_encoder_layer(network_helper, prefix, config, weights_dict, x, input_mask_tensor)

    x = network_helper.addTanh(encoder_out)

    return x

def build_aside(network_helper, weights_dict, tensor_list):
    # pre_encoder_layer_norm_bias (768,)
    # feature_emb_fc_b2 (384,)
    # multi_field_1 (11, 20)
    # multi_field_2 (11, 20)
    # feature_emb_fc_w2 (768, 384)

    # multi_field_7 (11, 20)
    # cls_out_b_aside (1,)

    # pre_encoder_layer_norm_scale (768,)
    # feature_emb_fc_w (160, 768)
    # multi_field_5 (11, 20)
    # multi_field_0 (1432, 20)
    # cls_out_w_aside (384, 1)
    # multi_field_4 (11, 20)
    # multi_field_6 (11, 20)
    # multi_field_3 (13, 20)
    # feature_emb_fc_b (768,)

    multi_field_0 = weights_dict["multi_field_0"]
    multi_field_1 = weights_dict["multi_field_1"]
    multi_field_2 = weights_dict["multi_field_2"]
    multi_field_3 = weights_dict["multi_field_3"]
    multi_field_4 = weights_dict["multi_field_4"]
    multi_field_5 = weights_dict["multi_field_5"]
    multi_field_6 = weights_dict["multi_field_6"]
    multi_field_7 = weights_dict["multi_field_7"]

    x0 = network_helper.addEmbedding(tensor_list[0], multi_field_0, "multi_field_0")
    x1 = network_helper.addEmbedding(tensor_list[1], multi_field_1, "multi_field_1")
    x2 = network_helper.addEmbedding(tensor_list[2], multi_field_2, "multi_field_2")
    x3 = network_helper.addEmbedding(tensor_list[3], multi_field_3, "multi_field_3")
    x4 = network_helper.addEmbedding(tensor_list[4], multi_field_4, "multi_field_4")
    x5 = network_helper.addEmbedding(tensor_list[5], multi_field_5, "multi_field_5")
    x6 = network_helper.addEmbedding(tensor_list[6], multi_field_6, "multi_field_6")
    x7 = network_helper.addEmbedding(tensor_list[7], multi_field_7, "multi_field_7")

    concat_tensors = [x0, x1, x2, x3, x4, x5, x6, x7]
    x = network_helper.addCat(concat_tensors, dim=1)

    x = network_helper.addShuffle(x, None, (-1, 1, 1, 160), None, "aside_reshape")

    feature_emb_fc_w = weights_dict["feature_emb_fc_w"]
    feature_emb_fc_b = weights_dict["feature_emb_fc_b"]
    x = network_helper.addLinear(x, feature_emb_fc_w, feature_emb_fc_b)

    x = network_helper.addReLU(x)

    # get output shape, used in another slice
    global slice_output_shape
    slice_output_shape = network_helper.network.add_shape(x).get_output(0)

    feature_emb_fc_w2 = weights_dict["feature_emb_fc_w2"]
    feature_emb_fc_b2 = weights_dict["feature_emb_fc_b2"]
    x = network_helper.addLinear(x, feature_emb_fc_w2, feature_emb_fc_b2)

    x = network_helper.addReLU(x)

    cls_out_w_aside = weights_dict["cls_out_w_aside"]
    cls_out_b_aside = weights_dict["cls_out_b_aside"]
    x = network_helper.addLinear(x, cls_out_w_aside, cls_out_b_aside)
    return x

def process_input_mask(network_helper, input_mask_tensor):
    # input_mask = input_mask.unsqueeze(-1)
    # attn_bias = input_mask.matmul(input_mask, transpose_y=True)
    # input_mask_tensor[B, S, 1]
    attn_bias = network_helper.addMatMul(input_mask_tensor, input_mask_tensor, False, True, "get attn_bias")

    # attn_bias[B, S, S]
    # attn_bias = (1. - attn_bias) * -10000.0
    # shape: [1, 1, 1]
    tmp_arr = np.array([[[-1.]]], dtype=np.float32)
    tmp_tensor = network_helper.addConstant(tmp_arr)
    attn_bias  = network_helper.addAdd(attn_bias, tmp_tensor)

    # tmp_arr = np.array([[[10000.]]], dtype=np.float32)
    # tmp_tensor = network_helper.addConstant(tmp_arr)
    attn_bias  = network_helper.addScale(attn_bias, 10000.0)
    # attn_bias = attn_bias.unsqueeze(1).tile([1, self.n_head, 1, 1])   # avoid broadcast =_=
    # attn_bias[B, 1, S, S]
    attn_bias = network_helper.addShuffle(attn_bias, None, (0, 1, 0, -1), None, "input_mask.unsqueeze(-1)")

    # network_helper.markOutput(attn_bias)

    return attn_bias

def build_model(network_helper, config, weights_dict, src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor, aside_tensor_list):
    #  def forward(self, x, labels=None):
        #  x, attn_weights = self.transformer(x)
        #  logits = self.head(x[:, 0])
    cls_aside_out = build_aside(network_helper, weights_dict, aside_tensor_list)
    # network_helper.markOutput(cls_aside_out)

    input_mask_tensor = process_input_mask(network_helper, input_mask_tensor)

    x = build_ernie_model(network_helper, config, weights_dict, src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor)
    # network_helper.markOutput(x)

    #  head.weight [10, 768]
    #  head.bias [10]
    cls_out_w = weights_dict["cls_out_w"]
    cls_out_b = weights_dict["cls_out_b"]
    import pdb; pdb.set_trace()
    cls_out = network_helper.addLinear(x, cls_out_w, cls_out_b)
    # print("cls_out")

    x = network_helper.addAdd(cls_out, cls_aside_out)
    # print("add cls_out and cls_aside_out")

    x = network_helper.addSigmoid(x)
    # print("sigmoid")
    # assert(0)

    return x

def build_engine(args, config, weights_dict, calibrationCacheFile):
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
        builder_config.max_workspace_size = args.workspace_size * (1024 * 1024)

        plugin_data_type:int = 0
        if args.fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
            plugin_data_type = 1

        if args.strict:
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        #  if args.use_strict:
            #  builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        network_helper = TrtNetworkHelper(network, plg_registry, TRT_LOGGER, plugin_data_type)

        # Create the network
        src_ids_tensor = network_helper.addInput(name="src_ids", dtype=trt.int32, shape=(-1, -1, 1))
        pos_ids_tensor = network_helper.addInput(name="pos_ids", dtype=trt.int32, shape=(-1, -1, 1))
        sent_ids_tensor = network_helper.addInput(name="sent_ids", dtype=trt.int32, shape=(-1, -1, 1))
        input_mask_tensor = network_helper.addInput(name="input_mask", dtype=trt.float32, shape=(-1, -1, 1))

        tmp6_tensor = network_helper.addInput(name="tmp6", dtype=trt.int32, shape=(-1, 1, 1))
        tmp7_tensor = network_helper.addInput(name="tmp7", dtype=trt.int32, shape=(-1, 1, 1))
        tmp8_tensor = network_helper.addInput(name="tmp8", dtype=trt.int32, shape=(-1, 1, 1))
        tmp9_tensor = network_helper.addInput(name="tmp9", dtype=trt.int32, shape=(-1, 1, 1))
        tmp10_tensor = network_helper.addInput(name="tmp10", dtype=trt.int32, shape=(-1, 1, 1))
        tmp11_tensor = network_helper.addInput(name="tmp11", dtype=trt.int32, shape=(-1, 1, 1))
        tmp12_tensor = network_helper.addInput(name="tmp12", dtype=trt.int32, shape=(-1, 1, 1))
        tmp13_tensor = network_helper.addInput(name="tmp13", dtype=trt.int32, shape=(-1, 1, 1))

        aside_tensor_list = [tmp6_tensor, tmp7_tensor, tmp8_tensor, tmp9_tensor, tmp10_tensor, tmp11_tensor, tmp12_tensor, tmp13_tensor]

        out = build_model(network_helper, config, weights_dict, src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor, aside_tensor_list)

        network_helper.markOutput(out)

        profile = builder.create_optimization_profile()
        min_shape = (1, 128, 1)
        opt_shape = (5, 128, 1)
        max_shape = (10, 128, 1)
        profile.set_shape("src_ids", min=min_shape, opt=opt_shape, max=max_shape)
        profile.set_shape("sent_ids", min=min_shape, opt=opt_shape, max=max_shape)
        profile.set_shape("pos_ids", min=min_shape, opt=opt_shape, max=max_shape)
        profile.set_shape("input_mask", min=min_shape, opt=opt_shape, max=max_shape)

        min_shape = (1, 1, 1)
        opt_shape = (5, 1, 1)
        max_shape = (10, 1, 1)
        profile.set_shape("tmp6", min=min_shape, opt=opt_shape, max=max_shape)
        profile.set_shape("tmp7", min=min_shape, opt=opt_shape, max=max_shape)
        profile.set_shape("tmp8", min=min_shape, opt=opt_shape, max=max_shape)
        profile.set_shape("tmp9", min=min_shape, opt=opt_shape, max=max_shape)
        profile.set_shape("tmp10", min=min_shape, opt=opt_shape, max=max_shape)
        profile.set_shape("tmp11", min=min_shape, opt=opt_shape, max=max_shape)
        profile.set_shape("tmp12", min=min_shape, opt=opt_shape, max=max_shape)
        profile.set_shape("tmp13", min=min_shape, opt=opt_shape, max=max_shape)
        builder_config.add_optimization_profile(profile)

        build_start_time = time.time()
        engine = builder.build_engine(network, builder_config)
        build_time_elapsed = (time.time() - build_start_time)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "build engine in {:.3f} Sec".format(build_time_elapsed))
        return engine

def load_paddle_weights(path_prefix):
    """
    Load the weights from the onnx checkpoint
    """

    exe = paddle.static.Executor(paddle.CPUPlace())
    # path_prefix = "./paddle_infer_model"

    [inference_program, feed_target_names, fetch_targets] = (
                paddle.static.load_inference_model(path_prefix, exe, model_filename="__model__", params_filename="__params__"))


    state_dict = inference_program.state_dict()

    print(feed_target_names)

    tensor_dict = {}
    for i in state_dict:
        # print(i)
        arr = np.array(state_dict[i])
        # print(arr.shape)

        tensor_dict[i] = arr

    TRT_LOGGER.log(TRT_LOGGER.INFO, "Load paddle model. Found {:} entries in weight map".format(len(tensor_dict)))

    return tensor_dict


def main():
    parser = argparse.ArgumentParser(description="TensorRT BERT Sample", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--paddle", required=True, help="The paddle model dir path.")
    parser.add_argument("-o", "--output", required=True, default="bert_base_384.engine", help="The bert engine file, ex bert.engine")
    parser.add_argument("-b", "--max_batch_size", default=1, type=int, help="max batch size")
    parser.add_argument("-f", "--fp16", action="store_true", help="Indicates that inference should be run in FP16 precision", required=False)
    parser.add_argument("-t", "--strict", action="store_true", help="Indicates that inference should be run in strict precision mode", required=False)
    parser.add_argument("-w", "--workspace-size", default=3000, help="Workspace size in MiB for building the BERT engine", type=int)
    # parser.add_argument("-n", "--calib-num", help="calibration cache path", required=False)

    args, _ = parser.parse_known_args()

    # calib_cache = "ViT_N{}L{}A{}CalibCache".format(args.model_type, config.transformer.num_layers, config.transformer.num_heads)
    # print(f"calib_cache = {calib_cache}")

    if args.paddle != None:
        weights_dict = load_paddle_weights(args.paddle)
    else:
        raise RuntimeError("You need either specify paddle using option --paddle to build TRT model.")

    # config.encoder_num_layers = 12
    # config.num_heads = 12
    config = []
    with build_engine(args, config, weights_dict, None) as engine:
        TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Serializing Engine...")
        serialized_engine = engine.serialize()
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(args.output))
        with open(args.output, "wb") as fout:
            fout.write(serialized_engine)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")

    # if args.img_path is not None:
    #     infer_helper = InferHelper(args.output, TRT_LOGGER)
    #     test_case_data(infer_helper, args, args.img_path)


if __name__ == "__main__":
    main()
