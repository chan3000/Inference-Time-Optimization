import tensorrt
import numpy as np
import os

USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32

BATCH_SIZE = 32


if USE_FP16:
    os.system('C:/Installations_Bharat/TensorRT/TensorRT-8.2.1.8.Windows10.x86_64.cuda-11.4.cudnn8.2/TensorRT-8.2.1.8/bin/trtexec.exe --onnx=parkingpytorchmodel.onnx --saveEngine=parkingpytorchmodel.trt --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16')
else:
    os.system('C:/Installations_Bharat/TensorRT/TensorRT-8.2.1.8.Windows10.x86_64.cuda-11.4.cudnn8.2/TensorRT-8.2.1.8/bin/trtexec.exe --onnx=parkingpytorchmodel.onnx --saveEngine=parkingpytorchmodel.trt --explicitBatch')

