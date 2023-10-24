import argparse
import numpy as np
import time
import ctypes

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt

def init_trt_plugin(severity=None, lib_name=None, logger=None):
    """
    TensorRT Initialization
    """
    if severity is None:
        severity = trt.Logger.INFO

    if logger is None:
        logger = trt.Logger(severity)

    lib_names = ["libnvinfer_plugin.so"]
    if lib_name is not None:
        lib_names.append(lib_name)
        # lib_name = "libtrt_plugin_plus.so"

    for lib in lib_names:
        handle = ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        if not handle:
            raise RuntimeError("Could not load plugin library. Is " + lib + " on your LD_LIBRARY_PATH?")

    trt.init_libnvinfer_plugins(logger, "")

    logger.log(logger.INFO, "[TrtHelper LOG] tensorrt plugin init done!")

    return logger

logger = init_trt_plugin(trt.Logger.VERBOSE)

class InferHelper():
    """"""
    def __init__(self, plan_name, trt_logger):
        """"""
        self.logger = trt_logger
        self.runtime = trt.Runtime(trt_logger)
        self.plan_name = plan_name
        with open(plan_name, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.context.active_optimization_profile = 0

    def infer(self, inputs: list, is_log = False):
        nInput = len(inputs)

        bufferD = []
        # alloc memory
        for i in range(nInput):
            bufferD.append(cuda.mem_alloc(inputs[i].nbytes))
            cuda.memcpy_htod(bufferD[i], inputs[i].ravel())
            self.context.set_binding_shape(i, tuple(inputs[i].shape))
            # print(inputs[i].nbytes)

        print("===============infer1 info===================")
        print("plan_name: " + self.plan_name)
        for i in range(0, self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.context.get_binding_shape(i)

            if self.engine.binding_is_input(i):
                print(f"input[{i}] name={name}, shape={shape}")
            else:
                print(f"output[{i}] name={name}, shape={shape}")
        print("=============================================")

        outputs = []
        for i in range(len(inputs), self.engine.num_bindings):
            outputs.append(np.zeros(self.context.get_binding_shape(i)).astype(np.float32))

        nOutput = len(outputs)
        for i in range(nOutput):
            bufferD.append(cuda.mem_alloc(outputs[i].nbytes))
            # print(outputs[i].nbytes)

        for i in range(len(inputs), self.engine.num_bindings):
            trt_output_shape = self.context.get_binding_shape(i)
            output_idx = i - len(inputs)
            if not (list(trt_output_shape) == list(outputs[output_idx].shape)):
                self.logger.log(trt.Logger.ERROR, "[Infer] output shape is error!")
                self.logger.log(trt.Logger.ERROR, "trt_output.shape = " + str(trt_output_shape))

        # warm up
        self.context.execute_v2(bufferD)

        T1 = time.perf_counter()

        self.context.execute_v2(bufferD)

        T2 =time.perf_counter()
        # if is_log:
        print("time=" + str((T2-T1) * 1000) + "ms")

        for i in range(nInput, nInput + nOutput):
            cuda.memcpy_dtoh(outputs[i - nInput].ravel(), bufferD[i])

        if is_log:
            for i in range(0, len(outputs)):
                print("outputs.shape:" + str(outputs[i].shape))
                print("outputs.sum:" + str(outputs[i].sum()))
                print(outputs[i])

            # print("trt_output.shape:" + str(trt_output.shape))
            # print("trt_output.sum:" + str(trt_output.sum()))
            # print(trt_output.view(-1)[0:10])
            # print("torch.allclose result:" + str(torch.allclose(base_output, trt_output, 1e-05, 1e-03)))
            # print("====================")
        return outputs
        # return torch.allclose(base_output, trt_output, 1e-05, 1e-03)

def cal_dif(onnx, trt, l):
    print("onnx_res:", onnx)
    print("trt_res: ", trt)
    dif_sum = 0
    min_dif = 1
    max_dif = 0
    for i in range(l):
        dif = abs(onnx[i] - trt[i]) / abs(onnx[i])
        max_dif = max(max_dif, dif)
        min_dif = min(min_dif, dif)
        dif_sum += dif
    return min_dif, max_dif, dif_sum

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
        res = np.array(res, dtype=np.int32)
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

            # # convert mask (-1, -1, 1) to (-1, 1)
            # mask = []
            # for i in range(s0):
                # sum = 0
                # for j in range(s1):
                    # for k in range(s2):
                        # sum += mask_data[i][j][k]
                # mask.append(sum)
            # mask = np.array(mask, dtype=np.int32).reshape(-1, 1)

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

            # print(src_data.shape, src_data.dtype)
            # print(pos_data.shape, pos_data.dtype)
            # print(sent_data.shape, sent_data.dtype)
            # print(mask.shape, mask.dtype)
            # print(aside1_data.shape, aside1_data.dtype)
            # print(aside2_data.shape, aside2_data.dtype)
            # print(aside3_data.shape, aside3_data.dtype)
            # print(aside4_data.shape, aside4_data.dtype)
            # print(aside5_data.shape, aside5_data.dtype)
            # print(aside6_data.shape, aside6_data.dtype)
            # print(aside7_data.shape, aside7_data.dtype)
            # print(aside8_data.shape, aside8_data.dtype)
            # assert(0)

            res.append([src_data, pos_data, sent_data, mask_data, aside1_data, aside2_data, aside3_data, aside4_data, aside5_data, aside6_data, aside7_data, aside8_data])
    return res

def main(args):
    # init infer_helper
    plan_name = args.plan_name
    file_name = args.input_file
    infer_helper = InferHelper(plan_name, logger)

    # 1 batch test data
    # src = [1,12,13,1557,40574,40997,22553,2,1571,40574,1569,42562,1557,40997,22553,1886,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # pos = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # sent = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # mask = [18]
    # aside = [1]
    # src = np.array(src,dtype=np.int32).reshape(1,128,1)
    # pos = np.array(pos,dtype=np.int32).reshape(1,128,1)
    # sent = np.array(sent,dtype=np.int32).reshape(1,128,1)
    # mask = np.array(mask,dtype=np.float32).reshape(1,128, 1)
    # aside = np.array(aside,dtype=np.int32).reshape(1,1,1)
    # print(src.shape, src.dtype)
    # print(pos.shape, pos.dtype)
    # print(sent.shape, sent.dtype)
    # print(mask.shape, mask.dtype)
    # print(aside.shape, aside.dtype)
    # inputs = [src, sent, pos, mask, aside, aside, aside, aside, aside, aside, aside, aside]

    # multi batch
    inputs = load_data(file_name)

    # # load onnx baseline
    # onnx_baseline = []
    # with open("./onnx_baseline.txt","r") as f:
        # lines = f.readlines()
        # for line in lines:
            # data = get_num(line, "float")
            # onnx_baseline.append(data)

    test_num = len(inputs)
    # test_num = 1

    # trt infer and compare with onnx baseline
    max_dif = 0
    min_dif = 1
    dif_sum = 0
    total_num = 0
    for i in range(test_num):
        output = infer_helper.infer(inputs[i], False)
        print(output[-1])
        if i == 5:
            break

        # b = onnx_baseline[i].shape[0]
        # print("infer and comparing case", i)
        # mindif, maxdif, dif = cal_dif(onnx_baseline[i].reshape(b), output[0].reshape(b), b)
        # max_dif = max(max_dif, maxdif)
        # min_dif = min(min_dif, mindif)
        # dif_sum += dif
        # total_num += b
    # print("min_dif:", min_dif, " max_dif:", max_dif, " avg_dif:", dif_sum / total_num)

    # speed test
    # warm up
    # output = infer_helper.infer(inputs[0], True)

    # start = time.perf_counter()
    # for i in range(test_num):
    #     output = infer_helper.infer(inputs[i], True)
    # end = time.perf_counter()

    # print("avg_time", (end-start) / test_num * 1000, "ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ernie trt model test")

    parser.add_argument("-p", "--plan_name", required=True, help="The trt plan file path.")
    parser.add_argument("-i", "--input_file", required=True, help="The onnx baseline file path.")

    args = parser.parse_args()
    main(args)
