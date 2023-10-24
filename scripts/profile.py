import numpy as np
import sys
import os
import subprocess

import ctypes
import argparse
# TensorRT
import tensorrt as trt

def init_trt_plugin(severity=None, lib_name=None, logger=None):
    """
    TensorRT Initialization
    """
    if severity is None:
        severity = trt.Logger.INFO

    if logger is None:
        logger = trt.Logger(severity)

    lib_names = ["libnvinfer_plugin.so", "libtrtplugin++.so"]
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

# logger = trt_helper.init_trt_plugin(trt.Logger.INFO, "libtorch_trt_plugin.so")
logger = init_trt_plugin(trt.Logger.VERBOSE)

def trt_dtype_to_np_dtype(trt_dtype):
    if trt_dtype == trt.DataType.FLOAT:
        return np.float32
    if trt_dtype == trt.DataType.INT32:
        return np.int32
    if trt_dtype == trt.DataType.HALF:
        return np.float16
    if trt_dtype == trt.DataType.INT8:
        return np.int8
    if trt_dtype == trt.DataType.BOOL:
        return np.bool_

def shape_to_str(shape):
    shape_len = len(shape)

    if shape_len == 1:
        return f"{shape[0]}"
    if shape_len == 2:
        return f"{shape[0]}x{shape[1]}"
    if shape_len == 3:
        return f"{shape[0]}x{shape[1]}x{shape[2]}"
    if shape_len == 4:
        return f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}"

class Profile():
    def __init__(self):
        self.plan_name = ""
        self.name_list = []
        self.nptype_list = []
        self.shape_list = []
        self.log_file = "profile.log"

    def profile(self, batch_size: int, seq_len: int):
        print(f"Start profile batch_size={batch_size} ====================================")
        all_tmp_filename = ""
        cmd = "./trtexec --loadEngine=" + self.plan_name
        cmd = cmd + " --plugins=/home/ubuntu/baidu_sti/Torch-TensorRT-Plugin/build/out/libtrtplugin++.so "

        shape_cmd_line = "--shapes="
        load_cmd_line = "--loadInputs="

        for i in range(len(self.name_list)):
            name = self.name_list[i]
            shape = self.shape_list[i]
            shape[0] = batch_size
            if i < 3:
                shape[1] = seq_len
            arr = np.zeros(shape, dtype=self.nptype_list[i])
            # if name == "feat_len":
            #     arr = np.ones(shape, dtype=self.nptype_list[i]) * self.shape_list[0][1]
            #     #  print(arr)
            #     #  assert 0
            print(shape)
            shape_str = shape_to_str(shape)
            # print(shape_str)
            tmp_filename = name + "_" + shape_str
            arr.tofile(tmp_filename)
            #  np.save(tmp_filename, arr)
            #  tmp_filename_list.append(tmp_filename_list)
            shape_cmd_line = shape_cmd_line + f"{name}:{shape_str},"
            load_cmd_line = load_cmd_line + f"\'{name}\':{tmp_filename},"

            all_tmp_filename = all_tmp_filename + tmp_filename + " "


        cmd = cmd + shape_cmd_line + " " + load_cmd_line
        #  print(cmd)
        os.system(cmd)
        #  log = subprocess.run(cmd, shell=True)
        #  with open(self.log_file, 'a+') as f:
            #  f.write(log)
        print(all_tmp_filename)
        os.system("rm " + all_tmp_filename)
        print(f"End profile batch_size={batch_size} ====================================")

def main(args):
    print(f"plan_name={args.plan}")
    runtime = trt.Runtime(logger)
    with open(args.plan, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        #  context.active_optimization_profile = 0

    p = Profile()
    p.plan_name = args.plan
    # dump info at first time

    for i in range(0, engine.num_bindings):
        if engine.binding_is_input(i):
            name = engine.get_binding_name(i)
            #  shape = context.get_binding_shape(i)
            trt_dtype = engine.get_binding_dtype(i)
            np_dtype = trt_dtype_to_np_dtype(trt_dtype)
            shape = list(context.get_binding_shape(i))

            p.name_list.append(name)
            p.nptype_list.append(np_dtype)
            p.shape_list.append(shape)

    p.profile(1,1)
    p.profile(5,64)
    p.profile(10,128)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trtexec profile")
    parser.add_argument('--plan', required=True, help='load path')
    # parser.add_argument('--batch', required=True, help='test read specifier file')

    args = parser.parse_args()
    main(args)

