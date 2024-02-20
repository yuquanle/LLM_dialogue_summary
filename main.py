import os
import torch
import json
import codecs
import argparse
from dataset import Dataset
from model import load_model
from torch.utils.data import DataLoader
from utils import build_elements_extract_prompt, build_elements_aware_summary_prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_idx', type=str, required=True)
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--input_path', type=str, default='input.json')
    parser.add_argument('--output_path', type=str, default='result.json')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    
    # load model.
    model, tokenizer = load_model(args.model_id)

    # load dataset.
    test_data = Dataset(test_file=args.input_path)

    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_data.collate_function)

    fw = codecs.open(args.output_path, "w", encoding="utf-8")

    # prompt template.
    # step1: elements extract prompt template.
    elements_extract_prompt_template = 'Please summarize the key elements from the following dialogue history, key elements include role information, topic, and import entities. \n Dialogue history: {dialogue_history_text}'
    # step2: elements-aware summary generation prompt template.
    elements_aware_summary_prompt_template = 'Please summarize the following dialogue summary with dialogue elements. \n Dialogue history: {dialogue_history_text} \n\n Dialogue elements: {dialogue_elements_text}'

    with torch.no_grad():
        for data in test_dataloader:    
            idx, dialogue_history, ground_truth_summary = data

            # step1: build elements extract prompt.
            input_elements_extract_prompt = [build_elements_extract_prompt(dialogue_text, elements_extract_prompt_template) for dialogue_text in dialogue_history]
            input_elements_extract = tokenizer(input_elements_extract_prompt, padding=True, return_tensors="pt", truncation=True, max_length=1024).to('cuda')

            elements_extract_outputs = model.generate(**input_elements_extract, do_sample=False, max_new_tokens=128, pad_token_id=2, eos_token_id=2)
            
            for idx in range(len(elements_extract_outputs)):
                output = elements_extract_outputs.tolist()[idx][len(input_elements_extract["input_ids"][idx]):]
                elements_output = tokenizer.decode(output)
           
            # step2: build elements-aware summary generation prompt. 
            input_elements_aware_summary_prompt = [build_elements_aware_summary_prompt(dialogue_text, dialogue_element, elements_aware_summary_prompt_template) for dialogue_text, dialogue_element in zip(dialogue_history, elements_output)]
            input_elements_aware_summary = tokenizer(input_elements_aware_summary_prompt, padding=True, return_tensors="pt", truncation=True, max_length=1024).to('cuda')

            elements_aware_summary_outputs = model.generate(**input_elements_aware_summary, do_sample=False, max_new_tokens=128, pad_token_id=2, eos_token_id=2)
            
            for idx in range(len(elements_aware_summary_outputs)):
                output = elements_aware_summary_outputs.tolist()[idx][len(input_elements_aware_summary["input_ids"][idx]):]
                preds_summary = tokenizer.decode(output)
            
            # write to output file.
            for pred_summary, gt_summary in zip(preds_summary, ground_truth_summary):
                fw.write(json.dumps({"preds_summary": pred_summary, "ground_truth_summary": gt_summary}, ensure_ascii=False) + "\n")
