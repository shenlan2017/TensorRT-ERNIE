
import numpy as np
import onnx
import onnxruntime
from onnx import helper

import argparse

def get_num(str, type):
    s = str.split(" ")
    res = []
    for ss in s:
        if type == "int":
            num = int(ss)
        else:
            num = float(ss)
        res.append(num)
    if type == "int":
        res = np.array(res, dtype=np.int64)
    else:
        res = np.array(res, dtype=np.float32)
    return res

def load_data(file_path):
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

            src_shape = get_num(src.split(":")[0], "int")
            s0 = src_shape[0]
            s1 = src_shape[1]
            s2 = src_shape[2]

            src_data = get_num(src.split(":")[1], "int").reshape(s0,s1,s2)
            pos_data = get_num(pos.split(":")[1], "int").reshape(s0,s1,s2)
            sent_data = get_num(sent.split(":")[1], "int").reshape(s0,s1,s2)
            mask_data = get_num(mask.split(":")[1], "float32").reshape(s0,s1,s2)

            aside_shape = get_num(aside1.split(":")[0], "int")
            s0 = aside_shape[0]
            s1 = aside_shape[1]
            s2 = aside_shape[2]

            aside1_data = get_num(aside1.split(":")[1], "int").reshape(s0,s1,s2)
            aside2_data = get_num(aside2.split(":")[1], "int").reshape(s0,s1,s2)
            aside3_data = get_num(aside3.split(":")[1], "int").reshape(s0,s1,s2)
            aside4_data = get_num(aside4.split(":")[1], "int").reshape(s0,s1,s2)
            aside5_data = get_num(aside5.split(":")[1], "int").reshape(s0,s1,s2)
            aside6_data = get_num(aside6.split(":")[1], "int").reshape(s0,s1,s2)
            aside7_data = get_num(aside7.split(":")[1], "int").reshape(s0,s1,s2)
            aside8_data = get_num(aside8.split(":")[1], "int").reshape(s0,s1,s2)

            res.append([src_data, pos_data, sent_data, mask_data, aside1_data, aside2_data, aside3_data, aside4_data, aside5_data, aside6_data, aside7_data, aside8_data])
    return res

def main(args):
    # init infer_helper
    onnx_name = args.onnx_name
    file_name = args.input_file

    # multi batch
    inputs = load_data(file_name)

    session = onnxruntime.InferenceSession(args.onnx_name)

    test_num = len(inputs)

    # infer and save onnx baseline
    for i in range(len(inputs)):
        input_list = {"read_file_0.tmp_0":inputs[i][0],
                      "read_file_0.tmp_1":inputs[i][1],
                      "read_file_0.tmp_2":inputs[i][2],
                      "read_file_0.tmp_3":inputs[i][3],
                      "read_file_0.tmp_6":inputs[i][4],
                      "read_file_0.tmp_7":inputs[i][5],
                      "read_file_0.tmp_8":inputs[i][6],
                      "read_file_0.tmp_9":inputs[i][7],
                      "read_file_0.tmp_10":inputs[i][8],
                      "read_file_0.tmp_11":inputs[i][9],
                      "read_file_0.tmp_12":inputs[i][10],
                      "read_file_0.tmp_13":inputs[i][11]}
        result = session.run(None, input_list)
        print(f"i=${i}, score=${result[0][0]}")
        if i == 5:
            break
        # res = result[0].reshape(1, result[0].shape[0])
        # np.savetxt(f, res, fmt='%.8f')
    # test_num = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ernie trt model test")

    parser.add_argument("-m", "--onnx_name", required=True, help="The onnx file path.")
    parser.add_argument("-i", "--input_file", required=True, help="The onnx baseline file path.")

    args = parser.parse_args()
    main(args)
