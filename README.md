# sface_trt
TensorRT sface

[TensorRT v8601] 

https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

trtexec --onnx=face_recognition_sface_2021dec.onnx --saveEngine=sface.trt

cmake -B build

cmake --build build

./build/main model/sface.trt -i test.jpg query.jpg 

