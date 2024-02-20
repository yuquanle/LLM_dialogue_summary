path_to_llama_model=/home/leyuquan/projects/LLMs/PLM_Backbones/chatglm3-6b-base
input_data=sft_demo_train_data
path_to_sft_checkpoint=./output/chatglm3-6b-base_lora_sft

CUDA_VISIBLE_DEVICES=3 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path ${path_to_llama_model} \
    --dataset ${input_data} \
    --template chatglm3 \
    --finetuning_type lora \
    --lora_target query_key_value \
    --output_dir ${path_to_sft_checkpoint} \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16