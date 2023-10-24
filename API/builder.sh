# fp32 build
#python3 API/builder.py -p ../sti2_data/model/paddle_infer_model -o trt_models/ernie_api_fp32.engine
python3 API/builder.py -p ../sti2_data/model/paddle_infer_model -o trt_models/ernie_api_fp32.engine
#python3 API/builder_base.py -p ../sti2_data/model/paddle_infer_model -o trt_models/ernie_api_fp32.engine

# fp16 build
#python ./src/builder.py --fp16 -p ./model/paddle_infer_model -o ./model/trt_model/ernie_fp16.engine

# int8 build
# python builder.py --strict --int8 -p ../model/paddle_infer_model -o ../model/trt_model/ernie_int8.engine -c /home/ubuntu/baidu_sti/model/calib_data/
