

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--gpu_idx=1 \
--model_id=chatglm3-6b \
--batch=2 \
--input_path=./data/demo_data.json \
--output_path=./result/demo_data_pred.json \