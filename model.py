import os
from transformers import AutoTokenizer, AutoModel


def load_model(model_id):
    print(f'Loading {model_id} model...')
    if model_id in ['chatglm3-6b']:
        model_path = os.path.join('/home/leyuquan/projects/LLMs/PLM_Backbones/chatglm3-6b/')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).bfloat16().cuda() 
    else:
        raise NameError
    print(f"load {model_id} model finish.")
    return model, tokenizer
