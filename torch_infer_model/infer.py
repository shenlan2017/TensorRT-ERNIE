# -*- coding: utf-8 -*

import logging
import json
import paddle
import torch
import numpy as np
import collections
import os
import pdb
import argparse

from ernie_model import Model
from ernie_config import *

def save_model(save_path):
    config = ErnieConfig()
    model = Model(config)
    torch.save(model.state_dict(), save_path)

def build_params_map(attention_num=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's BERT
    :return:
    """
    weight_map = collections.OrderedDict({
        'word_embedding': "ernie_model.word_emb.weight",
        'pos_embedding': "ernie_model.pos_emb.weight",
        'sent_embedding': "ernie_model.sent_emb.weight",
        'pre_encoder_layer_norm_scale': "ernie_model.ln.weight",
        'pre_encoder_layer_norm_bias': 'ernie_model.ln.bias',
    })
    # add attention layers
    for i in range(attention_num):  # paddle : torch
        weight_map[f'encoder_layer_{i}_multi_head_att_query_fc.w_0'] = f'ernie_model.encoder_stack.block.{i}.attn.q.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_query_fc.b_0'] = f'ernie_model.encoder_stack.block.{i}.attn.q.bias'
        weight_map[f'encoder_layer_{i}_multi_head_att_key_fc.w_0'] = f'ernie_model.encoder_stack.block.{i}.attn.k.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_key_fc.b_0'] = f'ernie_model.encoder_stack.block.{i}.attn.k.bias'
        weight_map[f'encoder_layer_{i}_multi_head_att_value_fc.w_0'] = f'ernie_model.encoder_stack.block.{i}.attn.v.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_value_fc.b_0'] = f'ernie_model.encoder_stack.block.{i}.attn.v.bias'
        weight_map[f'encoder_layer_{i}_multi_head_att_output_fc.w_0'] = f'ernie_model.encoder_stack.block.{i}.attn.o.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_output_fc.b_0'] = f'ernie_model.encoder_stack.block.{i}.attn.o.bias'
        weight_map[f'encoder_layer_{i}_post_att_layer_norm_scale'] = f'ernie_model.encoder_stack.block.{i}.ln1.weight'
        weight_map[f'encoder_layer_{i}_post_att_layer_norm_bias'] = f'ernie_model.encoder_stack.block.{i}.ln1.bias'
        weight_map[f'encoder_layer_{i}_ffn_fc_0.w_0'] = f'ernie_model.encoder_stack.block.{i}.ffn.i.weight'
        weight_map[f'encoder_layer_{i}_ffn_fc_0.b_0'] = f'ernie_model.encoder_stack.block.{i}.ffn.i.bias'
        weight_map[f'encoder_layer_{i}_ffn_fc_1.w_0'] = f'ernie_model.encoder_stack.block.{i}.ffn.o.weight'
        weight_map[f'encoder_layer_{i}_ffn_fc_1.b_0'] = f'ernie_model.encoder_stack.block.{i}.ffn.o.bias'
        weight_map[f'encoder_layer_{i}_post_ffn_layer_norm_scale'] = f'ernie_model.encoder_stack.block.{i}.ln2.weight'
        weight_map[f'encoder_layer_{i}_post_ffn_layer_norm_bias'] = f'ernie_model.encoder_stack.block.{i}.ln2.bias'
    # add pooler
    weight_map.update(
        {
            'pooled_fc.w_0': 'ernie_model.pooler.weight', # paddle : torch
            'pooled_fc.b_0': 'ernie_model.pooler.bias',
            'multi_field_0': 'aside.embed0.weight',
            'multi_field_1': 'aside.embed1.weight',
            'multi_field_2': 'aside.embed2.weight',
            'multi_field_3': 'aside.embed3.weight',
            'multi_field_4': 'aside.embed4.weight',
            'multi_field_5': 'aside.embed5.weight',
            'multi_field_6': 'aside.embed6.weight',
            'multi_field_7': 'aside.embed7.weight',
            'feature_emb_fc_w': 'aside.feature_emb_fc1.weight',
            'feature_emb_fc_b': 'aside.feature_emb_fc1.bias',
            'feature_emb_fc_w2': 'aside.feature_emb_fc2.weight',
            'feature_emb_fc_b2': 'aside.feature_emb_fc2.bias',
            'cls_out_w_aside': 'aside.cls_out.weight',
            'cls_out_b_aside': 'aside.cls_out.bias',
            'cls_out_w': 'cls_out.weight',
            'cls_out_b': 'cls_out.bias',
        }
    )
    return weight_map

def load_paddle_weights(paddle_path):
    """
    Load the weights from the onnx checkpoint
    """
    exe = paddle.static.Executor(paddle.CPUPlace())
    path_prefix = paddle_path
    paddle.enable_static()
    [inference_program, feed_target_names, fetch_targets] = (
                paddle.static.load_inference_model(path_prefix, exe, model_filename="__model__", params_filename="__params__"))

    state_dict = inference_program.state_dict()

    weight_dict = {}
    for i in state_dict:
        # print(i)
        arr = np.array(state_dict[i])
        # print(arr.shape)
        weight_dict[i] = torch.tensor(arr)
    return weight_dict

def extract_and_convert(paddle_path, torch_path):
    """
    :param input_dir:
    :param output_dir:
    :return:
    """
    config = ErnieConfig()
    state_dict = collections.OrderedDict()
    weight_map = build_params_map(attention_num=config['num_hidden_layers']) # paddle ：torch
    paddle_para = load_paddle_weights(paddle_path)  # paddle : para
    #assert(weight_map.keys() == paddle_para.keys())
    #pdb.set_trace()
    transpose_key = ['fc.w_0', 'fc_0.w_0', 'fc_1.w_0', 'dense.weight', 'feature_emb_fc_w', 'cls_out_w']
    for weight_name, weight_value in paddle_para.items():
        for key_point in transpose_key:
            if key_point in weight_name:
                weight_value = weight_value.t()
                break
        if weight_name not in weight_map:
            print('=' * 20, '[SKIP]', weight_name, '=' * 20)
            continue

        state_dict[weight_map[weight_name]] = torch.FloatTensor(weight_value)
        #print(weight_name, '->', weight_map[weight_name], weight_value.shape)
    torch.save(state_dict, os.path.join(torch_path))
    print("convert success!")

def get_num_label(str, type):
    s = str.split(" ")
    res = []
    for ss in s:
        if type == "int":
            num = int(ss)
        else:
            num = float(ss)
        res.append(num)
    if type == "int":
        res = np.array(res, dtype=np.int32)
    else:
        res = np.array(res, dtype=np.float32)
    return res

def load_data_label(file_path):
    res = []
    with open(file_path,"r") as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(";")
            src = data[2]
            pos = data[3]
            sent = data[4]
            mask = data[5]
            aside1 = data[6]
            aside2 = data[7]
            aside3 = data[8]
            aside4 = data[9]
            aside5 = data[10]
            aside6 = data[11]
            aside7 = data[12]
            aside8 = data[13]

            src_shape = get_num_label(src.split(":")[0], "int")
            s0 = src_shape[0]
            s1 = src_shape[1]
            s2 = src_shape[2]

            src_data = get_num_label(src.split(":")[1], "int").reshape(s0,s1)
            pos_data = get_num_label(pos.split(":")[1], "int").reshape(s0,s1)
            sent_data = get_num_label(sent.split(":")[1], "int").reshape(s0,s1)
            mask_data = get_num_label(mask.split(":")[1], "float32").reshape(s0,s1)


            aside_shape = get_num_label(aside1.split(":")[0], "int")
            s0 = aside_shape[0]
            s1 = aside_shape[1]
            s2 = aside_shape[2]

            aside1_data = get_num_label(aside1.split(":")[1], "int").reshape(s0,s1)
            aside2_data = get_num_label(aside2.split(":")[1], "int").reshape(s0,s1)
            aside3_data = get_num_label(aside3.split(":")[1], "int").reshape(s0,s1)
            aside4_data = get_num_label(aside4.split(":")[1], "int").reshape(s0,s1)
            aside5_data = get_num_label(aside5.split(":")[1], "int").reshape(s0,s1)
            aside6_data = get_num_label(aside6.split(":")[1], "int").reshape(s0,s1)
            aside7_data = get_num_label(aside7.split(":")[1], "int").reshape(s0,s1)
            aside8_data = get_num_label(aside8.split(":")[1], "int").reshape(s0,s1)

            res.append([src_data, pos_data, sent_data, mask_data, aside1_data, aside2_data, aside3_data, aside4_data, aside5_data, aside6_data, aside7_data, aside8_data])
    return res

def main(args):

    #load_paddle_weights(args.paddle_file)
    config = ErnieConfig()
    model = Model(config)
    if not os.path.exists(args.torch_model):
        extract_and_convert(args.paddle_file, args.torch_model)

    model.load_state_dict(torch.load(args.torch_model))
    input_datas = load_data_label(args.input_file)
    model.eval()
    for idx, data in enumerate(input_datas):
        src_ids_tensor = torch.from_numpy(data[0])
        pos_ids_tensor = torch.from_numpy(data[1])
        sent_ids_tensor = torch.from_numpy(data[2])
        input_mask_tensor = torch.from_numpy(data[3])

        tmp6_tensor = torch.from_numpy(data[4])
        tmp7_tensor = torch.from_numpy(data[5])
        tmp8_tensor = torch.from_numpy(data[6])
        tmp9_tensor = torch.from_numpy(data[7])
        tmp10_tensor = torch.from_numpy(data[8])
        tmp11_tensor = torch.from_numpy(data[9])
        tmp12_tensor = torch.from_numpy(data[10])
        tmp13_tensor = torch.from_numpy(data[11])

        aside_tensor_list = [tmp6_tensor, tmp7_tensor, tmp8_tensor, tmp9_tensor, tmp10_tensor, tmp11_tensor, tmp12_tensor, tmp13_tensor]

        with torch.no_grad():
            output = model(src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor, aside_tensor_list)
            print(output)
        if idx == 5:
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ernie trt model test")

    parser.add_argument("-p", "--paddle_file", default='./sti2_data/model/paddle_infer_model', type=str, help="The paddle model file path.")
    parser.add_argument("-i", "--input_file", default='./sti2_data/data/label.test.txt', type=str, help="The onnx baseline file path.")
    parser.add_argument("-t", "--torch_model", default='./ernie_torch.pth', type=str, help="torch model path")

    args = parser.parse_args()
    main(args)
