path_to_llama_model=/home/leyuquan/projects/LLMs/PLM_Backbones/chatglm3-6b-base
input_data=sft_demo_train_data
sft_model=chatglm3-6b-base_lora_sft
path_to_sft_checkpoint=./output/${sft_model}
path_to_predict_result=./results/${sft_model}_${input_data}

CUDA_VISIBLE_DEVICES=3 python src/train_bash.py \
--stage sft \
--do_predict \
--model_name_or_path ${path_to_llama_model} \
--adapter_name_or_path ${path_to_sft_checkpoint} \
--dataset ${input_data} \
--template chatglm3 \
--finetuning_type lora \
--output_dir ${path_to_predict_result} \
--per_device_eval_batch_size 8 \
--predict_with_generate \
--fp16

