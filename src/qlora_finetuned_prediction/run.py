
import re
import math
import torch
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
import matplotlib.pyplot as plt
import pickle
import os
from src.config import HF_USER
from src.tester import Tester
from src.data_curation.item import Item
from peft import PeftModel

def run():
    # Constants
    BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
    RUN_NAME = "2025-08-30_15.24.25"
    PROJECT_RUN_NAME = f"alistermarc-2025-08-30_15.24.25"
    FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"
    DATASET_NAME = f"alistermarc/llama3-pricer-2025-08-30_02.01.02"

    # Log in to HuggingFace
    from dotenv import load_dotenv
    load_dotenv(override=True)
    hf_token = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')
    login(hf_token, add_to_git_credential=True)

    dataset = load_dataset(DATASET_NAME)
    train = dataset['train']
    test = dataset['test']

    # # Load Dataset
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # with open(os.path.join(project_root, 'data', 'train.pkl'), 'rb') as file:
    #     train = pickle.load(file)

    # with open(os.path.join(project_root, 'data', 'test.pkl'), 'rb') as file:
    #     test = pickle.load(file)

    # Load Tokenizer and Quantized LLaMA Model
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

    print(f"Memory footprint: {fine_tuned_model.get_memory_footprint() / 1e9:.1f} GB")

    # Prediction
    def extract_price(s):
        if "Price is $" in s:
          contents = s.split("Price is $")[1]
          contents = contents.replace(',','').replace('$','')
          match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
          return float(match.group()) if match else 0
        return 0

    def model_predict(item):
        prompt = item.test_prompt()
        set_seed(42)
        inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape, device="cuda")
        outputs = fine_tuned_model.generate(inputs, max_new_tokens=4, attention_mask=attention_mask, num_return_sequences=1)
        response = tokenizer.decode(outputs[0])
        return extract_price(response)

    # Run Evaluation
    Tester.test(model_predict)
