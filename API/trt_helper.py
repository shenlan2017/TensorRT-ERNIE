#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import torch
import tensorrt as trt
import numpy as np
import ctypes
import math
import time

from typing import Optional, Tuple

def set_tensor_name(tensor, prefix, name):
    tensor.name = prefix + name

def set_output_name(layer, prefix, name, out_idx = 0):
    set_tensor_name(layer.get_output(out_idx), prefix, name)

def set_output_range(layer, maxval, out_idx = 0):
    layer.get_output(out_idx).set_dynamic_range(-maxval, maxval)

def get_mha_dtype(args):
    dtype = trt.float32
    if args.fp16:
        dtype = trt.float16
    # Multi-head attention doesn't use INT8 inputs and output by default unless it is specified.
    # if config.int8 and config.use_int8_multihead and not config.is_calib_mode:
    #     dtype = trt.int8
    return int(dtype)

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

class TrtNetworkHelper():
    """TensorRT Network Definition helper for Pytorch"""
    def __init__(self, network, plugin_registry, logger, plugin_data_type):
        self.network = network
        self.plugin_registry = plugin_registry
        self.logger = logger

        self.input_num = 0

        self.np_data_type = np.array([plugin_data_type], dtype=np.int32)

    def broadcast_matrix(self, mat: np.array, nb_dims: int):
        mat_nb_dims = len(mat.shape)
        if mat_nb_dims >= nb_dims:
            raise RuntimeError("broadcast_tensor mat_nb_dims >= nb_dims")

        new_shape = np.ones([nb_dims], dtype=np.int32)
        new_shape[-mat_nb_dims:] = mat.shape

        new_mat = mat.reshape(new_shape)
        self.logger.log(trt.Logger.INFO, "[Network] broadcast_matrix " + \
                                          str(mat.shape) + " to " + str(new_mat.shape))

        return new_mat

    def set_layer_name(self, layer, name):
        """
        Tool function. Set the name of trt layer or plugin and print output shapes.
        """
        if not layer:
            raise RuntimeError("Could not name")

        layer.name = str(self.network.num_layers) + "_" + name
        for i in range(0, layer.num_outputs):
            shape = layer.get_output(i).shape
            self.logger.log(trt.Logger.INFO, "[Network] " + layer.name + ", output[" + str(i) + "] shape= " + str(shape))

        return None

    def check_trt_layer(self, trt_layer):
        """
        Tool function. check trt layer,
        """
        if not trt_layer:
            raise RuntimeError("add " + str(trt_layer) + " failed!")

        for i in range(0, trt_layer.num_outputs):
            shape = trt_layer.get_output(i).shape
            # print(trt.volume(shape))

            # if len(shape) is 1:
                # raise RuntimeError("add " + layer.name + " failed!")

    def layer_post_process(self, trt_layer, layer_name, precision):
        """
        Tool function. set precision, set_layer_name and check_trt_layer
        """
        if precision is not None:
            trt_layer.precision = precision

        self.set_layer_name(trt_layer, layer_name)
        self.check_trt_layer(trt_layer)

    def addInput(self, name, dtype, shape):
        if name is None:
            name = "input" + str(self.input_num)

        self.input_num = self.input_num + 1

        trt_input = self.network.add_input(name=name, dtype=dtype, shape=shape)
        if not trt_input:
            raise RuntimeError("addInput failed!")

        self.logger.log(trt.Logger.INFO, "[Network] add input:" + name + ", shape=" + str(shape))

        return trt_input

    def markOutput(self, x: trt.ITensor):
        self.network.mark_output(x)
        self.logger.log(trt.Logger.INFO, "[Network] mark output:" + x.name + ", shape=" + str(x.shape))

    def addConv2d(self, x, weight, bias, out_channels, kernel_size, stride=None, padding=None, dilation=None, groups=None,
                  layer_name=None, precision=None):
        """ConvCommon"""
        if layer_name is None:
            layer_name = "nn.Conv2d"
        else:
            layer_name = "nn.Conv2d." + layer_name

        weight = trt.Weights(weight)
        bias = trt.Weights(bias) if bias is not None else None

        trt_layer = self.network.add_convolution_nd(
            x, num_output_maps=out_channels,
            kernel_shape=kernel_size,
            kernel=weight, bias=bias)

        if stride is not None:
            trt_layer.stride = stride
        if padding is not None:
            trt_layer.padding = padding
        if dilation is not None:
            trt_layer.dilation = dilation
        if groups is not None:
            trt_layer.num_groups = groups

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addDumpTensor(self, x: trt.ITensor, layer_name: str = None):
        """DumpTensorPlugin"""
        plg_creator = self.plugin_registry.get_plugin_creator("DumpTensorPluginDynamic", "1", "")
        if not plg_creator:
            raise RuntimeError("Could not find DumpTensorPluginDynamic")

        if layer_name is None:
            layer_name = "DumpTensorPlugin"
        else:
            layer_name = "DumpTensorPlugin." + layer_name

        # data_type = trt.PluginField("data_type", np.array([data_type], dtype=np.int32), trt.PluginFieldType.INT32)
        # pfc = trt.PluginFieldCollection([data_type])
        pfc = trt.PluginFieldCollection([])
        plugin = plg_creator.create_plugin(layer_name, pfc)
        if not plugin:
            raise RuntimeError("Could not create_plugin DumpTensorPluginDynamic")

        layer = self.network.add_plugin_v2([x], plugin)

        self.layer_post_process(layer, layer_name, None)

        x = layer.get_output(0)
        return x

    def addEmbedding(self, indices, weight, layer_name=None, precision=None):
        constant_layer = self.network.add_constant(weight.shape, trt.Weights(weight))
        gather_layer = self.network.add_gather(constant_layer.get_output(0),
                                               indices, axis=0)

        if layer_name is None:
            layer_name = "nn.Embedding"
        else:
            layer_name = "nn.Embedding." + layer_name

        self.layer_post_process(gather_layer, layer_name, precision)

        return gather_layer.get_output(0)

    def addGELU(self, x, layer_name=None, precision=None):
        POW = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
        MULTIPLY = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
        SQRT = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
        ONE = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
        HALF = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
        X_pow = self.network.add_elementwise(x, POW.get_output(0), trt.ElementWiseOperation.POW)
        X_pow_t = X_pow.get_output(0)
        X_mul = self.network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
        X_add = self.network.add_elementwise(x, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
        X_sqrt = self.network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
        X_sqrt_tensor = X_sqrt.get_output(0)
        X_tanh = self.network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
        X_tanh_tensor = X_tanh.get_output(0)
        X_one = self.network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
        CDF = self.network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
        gelu_layer = self.network.add_elementwise(CDF.get_output(0), x, trt.ElementWiseOperation.PROD)

        if layer_name is None:
            layer_name = "nn.GELU"
        else:
            layer_name = "nn.GELU." + layer_name

        self.layer_post_process(gelu_layer, layer_name, precision)

        return gelu_layer.get_output(0)

    # def addSigmoid(self, layer, x, layer_name=None, precision=None):
    def addSigmoid(self, x, layer_name=None, precision=None):
        """Sigmoid"""
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.SIGMOID)

        if layer_name is None:
            layer_name = "nn.Sigmoid"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    # def addLayerNorm(self, layer, x, layer_name=None, precision=None):
    def addLayerNorm(self, x, weight, bias, layer_name=None, precision=None):
        """LayerNorm"""
        plg_creator = self.plugin_registry.get_plugin_creator("LayerNorm", "1", "")
        if not plg_creator:
            raise RuntimeError("Could not find LayerNorm")

        # dim = layer.weight.size(0)
        # eps = layer.eps
        # gamma = layer.weight
        # beta = layer.bias
        dim = 768
        eps = 0.00001
        gamma = weight
        beta = bias
        # data_type = trt.PluginField("data_type", self.np_data_type, trt.PluginFieldType.INT32)
        # dim = trt.PluginField("dim", np.array([dim], dtype=np.int32), trt.PluginFieldType.INT32)
        # eps = trt.PluginField("eps", np.array([eps], dtype=np.float32), trt.PluginFieldType.FLOAT32)
        # gamma_w = trt.PluginField("gamma", gamma.detach().numpy(), trt.PluginFieldType.FLOAT32)
        # beta_w = trt.PluginField("beta", beta.detach().numpy(), trt.PluginFieldType.FLOAT32)
        pfc = trt.PluginFieldCollection([])
        plugin = plg_creator.create_plugin("LayerNormPluginDynamic", pfc)
        if not plugin:
            raise RuntimeError("Could not create_plugin LayerNormPluginDynamic")

        gamma_w = self.addConstant(gamma)
        beta_w = self.addConstant(beta)
        trt_layer = self.network.add_plugin_v2([x, gamma_w, beta_w], plugin)

        if layer_name is None:
            layer_name = "nn.LayerNorm"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addLinear(self, x, weight, bias=None, layer_name=None, precision=None):
        """Linear"""
        # If input B is a constant, we transpose at parse time if necessary,
        # because In some cases, A * Bt is much slower than A * B.
        # weight = np.copy(weight.transpose(1, 0), order='C')
        weight = self.broadcast_matrix(weight, len(x.shape))

        weight_layer = self.network.add_constant(weight.shape, trt.Weights(weight))
        weight = weight_layer.get_output(0)
        # trt_layer = self.network.add_matrix_multiply(x, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.TRANSPOSE)
        trt_layer = self.network.add_matrix_multiply(x, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.NONE)
        x = trt_layer.get_output(0)

        if layer_name is None:
            layer_name = "Linear"
        else:
            layer_name = "Linear." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        if bias is not None:
            bias = self.broadcast_matrix(bias, len(x.shape))
            bias_layer = self.network.add_constant(bias.shape, trt.Weights(bias))
            bias = bias_layer.get_output(0)
            trt_layer = self.network.add_elementwise(x, bias, trt.ElementWiseOperation.SUM)
            x = trt_layer.get_output(0)

            if layer_name is None:
                layer_name = "Linear.bias"
            else:
                layer_name = "Linear.bias." + layer_name

        return x

    def addReshape(self, x, reshape_dims, layer_name=None, precision=None):
        trt_layer = self.network.add_shuffle(x)
        trt_layer.reshape_dims = reshape_dims

        if layer_name is None:
            layer_name = "torch.reshape"
        else:
            layer_name = "torch.reshape." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)

        x = trt_layer.get_output(0)
        return x

    def addSlice(self, x, start_dim, shape_dim, stride_dim, layer_name=None, precision=None):
        trt_layer = self.network.add_slice(x, start_dim, shape_dim, stride_dim)

        if layer_name is None:
            layer_name = "tensor.slice"
        else:
            layer_name = "tensor.slice." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)

        x = trt_layer.get_output(0)
        return x

    def addReLU(self, x, layer_name=None, precision=None):
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.RELU)

        if layer_name is None:
            layer_name = "nn.ReLU"

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addTanh(self, x, layer_name=None, precision=None):
        """Tanh"""
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.TANH)

        if layer_name is None:
            layer_name = "nn.Tanh"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    ################## unary op ###################
    def addLog(self, x: trt.ITensor, layer_name=None, precision=None):
        trt_layer = self.network.add_unary(x, trt.UnaryOperation.LOG)
        if layer_name is None:
            layer_name = "unary.log"
        else:
            layer_name = "unary.log." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    ################## elementwise op ###################
    def addAdd(self, a, b, layer_name=None, precision=None):
        trt_layer = self.network.add_elementwise(a, b, trt.ElementWiseOperation.SUM)
        if layer_name is None:
            layer_name = "elementwise.sum"
        else:
            layer_name = "elementwise.sum." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    # tensor and scalar op
    def addScale(
            self,
            x: trt.ITensor,
            scale: float,
            layer_name: str = None,
            precision: trt.DataType = None
    ) -> trt.ITensor:
        """scale"""
        input_len = len(x.shape)
        if input_len < 3:
            raise RuntimeError("input_len < 3 not support now! ")

        if layer_name is None:
            layer_name = "Scale"

        # The input dimension must be greater than or equal to 4
        if input_len is 3:
            trt_layer = self.network.add_shuffle(x)
            trt_layer.reshape_dims = (0, 0, 0, 1)
            self.layer_post_process(trt_layer, layer_name+".3dto4d", precision)
            x = trt_layer.get_output(0)

        np_scale = trt.Weights(np.array([scale], dtype=np.float32))
        trt_layer = self.network.add_scale(x, mode=trt.ScaleMode.UNIFORM,
                                      shift=None, scale=np_scale, power=None)
        self.layer_post_process(trt_layer, layer_name, precision)
        x = trt_layer.get_output(0)

        if input_len is 3:
            trt_layer = self.network.add_shuffle(x)
            trt_layer.reshape_dims = (0, 0, 0)
            self.layer_post_process(trt_layer, layer_name+".4dto3d", precision)
            x = trt_layer.get_output(0)

        return x

    def addSoftmax(self, x: trt.ITensor, dim: int = -1, layer_name=None, precision=None) -> trt.ITensor:
        trt_layer = self.network.add_softmax(x)

        input_len = len(x.shape)
        if input_len < 2:
            raise RuntimeError("softmax input_len must >= 2")

        if dim < 0:
            dim = input_len + dim

        trt_layer.axes = 1 << dim

        layer_name_prefix = "nn.Softmax[dim=" + str(dim) + "]"
        if layer_name is None:
            layer_name = layer_name_prefix
        else:
            layer_name = layer_name_prefix + "." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addCat(self, inputs = [], dim = 0, layer_name=None, precision=None):
        assert len(inputs) > 1

        trt_layer = self.network.add_concatenation(inputs)

        if dim == -1:
            dim = len(inputs[0].shape) - 1

        trt_layer.axis = dim

        if layer_name is None:
            layer_name = "torch.cat"
        else:
            layer_name = "torch.cat." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addMatMul(self, a, b, trans_a=False, trans_b=False, layer_name=None, precision=None):
        """Matmul"""
        if layer_name is None:
            layer_name = "matrix_multiply"
        else:
            layer_name = "matrix_multiply." + layer_name

        op_a = trt.MatrixOperation.NONE
        if trans_a is True:
            op_a = trt.MatrixOperation.TRANSPOSE

        op_b = trt.MatrixOperation.NONE
        if trans_b is True:
            op_b = trt.MatrixOperation.TRANSPOSE

        trt_layer = self.network.add_matrix_multiply(a, op_a, b, op_b)
        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addConstant(self, w, layer_name: Optional[str] = None) -> trt.ITensor:
        trt_layer = self.network.add_constant(w.shape, w)

        if layer_name is None:
            layer_name = "trt.Constant"
        else:
            layer_name = "trt.Constant." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)
        x = trt_layer.get_output(0)
        return x

    def addShuffle(
        self,
        x: trt.ITensor,
        first_transpose: trt.Permutation,
        reshape_dims: trt.Dims,
        second_transpose: trt.Permutation,
        layer_name: Optional[str] = None
    ) -> trt.ITensor:
        """"""
        trt_layer = self.network.add_shuffle(x)
        if first_transpose is not None:
            trt_layer.first_transpose = first_transpose

        if reshape_dims is not None:
            trt_layer.reshape_dims = reshape_dims

        if second_transpose is not None:
            trt_layer.second_transpose = second_transpose

        if layer_name is None:
            layer_name = "trt.Shuffle"
        else:
            layer_name = "trt.Shuffle." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)

        x = trt_layer.get_output(0)
        return x


