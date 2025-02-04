# sface_trt
TensorRT sface

[TensorRT v8601] 


$ trtexec --onnx=face_recognition_sface_2021dec.onnx --saveEngine=sface.trt

$ ./build/main model/sface.trt -i test.jpg query.jpg 

