#export LD_LIBRARY_PATH=../so:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=../so/out:$LD_LIBRARY_PATH

echo $LD_LIBRARY_PATH
# fp32
python scripts/trt_infer.py -p trt_models/ernie_api_fp32.engine -i data/label.test.txt
#python scripts/trt_infer.py -p trt_models/ernie_onnx_fp32.engine -i data/label.test.txt

# fp16
#python trt_infer.py -p ../model/trt_model/ernie_fp16.engine -i ../data/perf.test.txt

# int8
# python trt_infer.py -p /home/ubuntu/baidu_sti/model/trt_model/ernie_int8.engine -i /home/ubuntu/sti2_data/data/perf.test.txt
