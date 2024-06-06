python3 eval_power.py --img_size 16 --savepath result
python3 eval_power.py --img_size 32 --savepath result
python3 eval_power.py --img_size 64 --savepath result
python3 eval_power.py --img_size 128 --savepath result

python3 eval_latency.py --img_size 16 --savepath result
python3 eval_latency.py --img_size 32 --savepath result
python3 eval_latency.py --img_size 64 --savepath result
python3 eval_latency.py --img_size 128 --savepath result

python3 create_result.py