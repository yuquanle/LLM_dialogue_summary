# Project of element-aware summarization with LLM.

## Introduction
This project includes: 
- directly use LLM for dialogue summary tasks.
- use instruction fine-tuning to align dialogue summary tasks.

## Directly use LLM for dialogue summary

```
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--gpu_idx=0 \
--model_id=chatglm3-6b \
--batch=1 \
--input_path=./data/demo_data.json \
--output_path=./result/demo_data_pred.json \
```
This example uses chatglm3-6b as encoder. You can refer this code to implement other LLMs, just replace the model's load code.

## Use instruction fine-tuning to align dialogue summary 

Due to the fact that the model may not have seen these new instructions, it may be difficult to align them, so we can construct the instruction alignment tasks to align the model.

This project involves two instruction alignment tasks, as shown in the following examples:

- key elements extract task: 
```
{
"instruction": "Please summarize the key elements from the following dialogue history, key elements include role information, topic, and import entities.",
"input": "Dialogue history: #Speaker_1#: Good morning. What can I do for you?\n #Speaker_2#: I’m in Room 309. I’m checking out today. Can I have my bill now?\n #Speaker_1#: Certainly. Please wait a moment. Here you are.\n #Speaker_2#: Thanks. Wait…What’s this? The 30 dollar for?\n #Speaker_1#: Excuse me… The charge for your laundry service on Nov. 20th.\n #Speaker_2#: But I didn’t take any laundry service during my stay here. I think you have added someone else’s.\n #Speaker_1#: Ummm… Sorry, would you mind waiting amoment? We check it with the department concerned.\n #Speaker_2#: No. As long as we get this straightened out.\n #Speaker_1#: I’m very sorry. There has been a mistake. We’ll corrected the bill. Please take a look.\n #Speaker_2#: Okay, here you are.\n #Speaker_1#: Goodbye",
"output": "Dialogue role includes #Speaker_1# and #Speaker_2#. Topic is .... Important entities include....."
}
```
- element-aware dialogue summary task:
```
{
"instruction": "Please summarize the following dialogue summary with dialogue elements.",
"input": "Dialogue history: #Speaker_1#: Good morning. What can I do for you?\n #Speaker_2#: I’m in Room 309. I’m checking out today. Can I have my bill now?\n #Speaker_1#: Certainly. Please wait a moment. Here you are.\n #Speaker_2#: Thanks. Wait…What’s this? The 30 dollar for?\n #Speaker_1#: Excuse me… The charge for your laundry service on Nov. 20th.\n #Speaker_2#: But I didn’t take any laundry service during my stay here. I think you have added someone else’s.\n #Speaker_1#: Ummm… Sorry, would you mind waiting amoment? We check it with the department concerned.\n #Speaker_2#: No. As long as we get this straightened out.\n #Speaker_1#: I’m very sorry. There has been a mistake. We’ll corrected the bill. Please take a look.\n #Speaker_2#: Okay, here you are.\n #Speaker_1#: Goodbye \n\n Dialogue elements: ...",
"output": "#Speaker_1# helps #Speaker_2# correct a mischarged bill on laundry service and helps #Person_2# check out."
}
```

We need to annotate a small number (perhaps hundreds) of instruction fine-tuning samples. In annotation, dialogue elements need to be defined based on actual scenarios.

Next, the LLaMA-Factory framework can be used to instruct fine-tuning of the model, with the following commands:
```
path_to_llama_model=/home/leyuquan/projects/LLMs/PLM_Backbones/chatglm3-6b-base
input_data=sft_demo_train_data
path_to_sft_checkpoint=./output/chatglm3-6b-base_lora_sft

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
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
```


After fine-tuning, the following code can be used for prediction:
```
path_to_llama_model=/home/leyuquan/projects/LLMs/PLM_Backbones/chatglm3-6b-base
input_data=sft_demo_train_data
sft_model=chatglm3-6b-base_lora_sft
path_to_sft_checkpoint=./output/${sft_model}
path_to_predict_result=./results/${sft_model}_${input_data}

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
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
```

This example uses chatglm3-6b for instruction fine-tuning. LLaMA-Factory framework also supports other popular LLM, such as BLOOM and LLaMA. You can refer this link (https://github.com/hiyouga/LLaMA-Factory/tree/main) for detail.

