import os
import cv2 as cv


import pycuda.driver as cuda
import pycuda.autoinit
import cv2 as cv
import numpy as np
import os
import tensorrt as trt

import time

TRT_LOGGER = trt.Logger()
model_path = "parkingpytorchmodel.onnx" # .onnx file
engine_file_path = "parkingpytorchmodel.trt"	# .trt file

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    start = time.time()
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    end = time.time()
    timeRec = end - start

    return [out.host for out in outputs], float(timeRec)
        
with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    total_time = 0
    count = 0
    
    TEST_DIR = "test_data/All"	# Enter the test directory
    test_imgs_path = []
    for img_path in os.listdir(TEST_DIR):
        test_imgs_path.append(os.path.join(TEST_DIR, img_path))
    size = float(len(test_imgs_path))
    for img_path in test_imgs_path:
        
        if count > 0 and count % 91 == 0:
                print(f"Time Taken | Batch {int(count/91)}  = " + str(total_time) + " seconds")
                total_time = 0
        count += 1
        image = cv.imread(img_path)
        image = cv.resize(image,(150, 150))
        image=image[np.newaxis,np.newaxis, :, :].astype(np.float16)
        
        inputs[0].host = image
        do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        trt_outputs, time_inf = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        print(trt_outputs)
        total_time += time_inf
        print("The time taken for inference is: " + str(time_inf) + " seconds")

    print("The mean time taken for inference is: " + str(total_time/size) + " seconds")

        
