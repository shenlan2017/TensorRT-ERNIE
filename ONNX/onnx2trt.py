import tensorrt as trt
import os

import numpy as np
import onnx

import torch
from torch.nn import functional as F
import numpy as np
import os
import argparse

def onnx2trt(args):
    onnx_file = args.model
    logger = trt.Logger(trt.Logger.VERBOSE)

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config.max_workspace_size = 3<<30

    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnx_file):
        raise RuntimeError("Failed finding onnx file!")

    print("Succeeded finding onnx file!")
    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
    print("Succeeded parsing ONNX file!")

    # Create the network
    src_ids_tensor = network.get_input(0)
    pos_ids_tensor = network.get_input(1)
    sent_ids_tensor = network.get_input(2)
    input_mask_tensor = network.get_input(3)

    tmp6_tensor = network.get_input(4)
    tmp7_tensor = network.get_input(5)
    tmp8_tensor = network.get_input(6)
    tmp9_tensor = network.get_input(7)
    tmp10_tensor = network.get_input(8)
    tmp11_tensor = network.get_input(9)
    tmp12_tensor = network.get_input(10)
    tmp13_tensor = network.get_input(11)

    profile = builder.create_optimization_profile()
    min_shape = (1, 128, 1)
    opt_shape = (5, 128, 1)
    max_shape = (10, 128, 1)
    profile.set_shape(src_ids_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
    profile.set_shape(pos_ids_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
    profile.set_shape(sent_ids_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
    profile.set_shape(input_mask_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)

    min_shape = (1, 1, 1)
    opt_shape = (5, 1, 1)
    max_shape = (10, 1, 1)
    profile.set_shape(tmp6_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
    profile.set_shape(tmp7_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
    profile.set_shape(tmp8_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
    profile.set_shape(tmp9_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
    profile.set_shape(tmp10_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
    profile.set_shape(tmp11_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
    profile.set_shape(tmp12_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
    profile.set_shape(tmp13_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    if not engine:
        raise RuntimeError("build_engine failed")
    print("Succeeded building engine!")

    print("Serializing Engine...")
    serialized_engine = engine.serialize()
    if serialized_engine is None:
        raise RuntimeError("serialize failed")

    with open(args.output, "wb") as fout:
        fout.write(serialized_engine)

def trt_infer(plan_name):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."
    encoded_input = tokenizer.encode_plus(text, return_tensors = "pt")
    mask_index = torch.where(encoded_input["input_ids"][0] == tokenizer.mask_token_id)
    print(encoded_input)

    """
    TensorRT Initialization
    """
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    # # TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
    # if not handle:
        # raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.so` on your LD_LIBRARY_PATH?")

    infer_helper = InferHelper(plan_name, TRT_LOGGER)
    # input_list = [encoded_input['input_ids'].detach().numpy(), encoded_input['attention_mask'].detach().numpy(), encoded_input['token_type_ids'].detach().numpy()]
    input_list = [encoded_input['input_ids'].detach().numpy(), encoded_input['token_type_ids'].detach().numpy(), encoded_input['attention_mask'].detach().numpy()]

    output = infer_helper.infer(input_list)
    print(output)

    logits = torch.from_numpy(output[0])
    softmax = F.softmax(logits, dim = -1)
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
    print(top_10)
    print("model test topk10 output:")
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(new_sentence)

    print("hhhh, the result is wrong, debug yourself")

def main():
    parser = argparse.ArgumentParser(description="TensorRT Ernie Sample", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", required=True, help="The onnx model dir path.")
    parser.add_argument("-o", "--output", required=True, default="bert_base_384.engine", help="The bert engine file, ex bert.engine")
    parser.add_argument("-b", "--max_batch_size", default=1, type=int, help="max batch size")
    parser.add_argument("-f", "--fp16", action="store_true", help="Indicates that inference should be run in FP16 precision", required=False)
    parser.add_argument("-i", "--int8", action="store_true", help="Indicates that inference should be run in INT8 precision", required=False)
    parser.add_argument("-t", "--strict", action="store_true", help="Indicates that inference should be run in strict precision mode", required=False)
    parser.add_argument("-w", "--workspace-size", default=3000, help="Workspace size in MiB for building the BERT engine", type=int)
    parser.add_argument("-c", "--calib_path", help="calibration cache path", required=False)

    args, _ = parser.parse_known_args()

    onnx2trt(args)

if __name__ == '__main__':
    main()
    # trt_infer(plan_name)


