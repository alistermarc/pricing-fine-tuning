import pickle
from src.tester import Tester

def calculate_average_price():
    # with open("data/train.pkl", "rb") as f:
    #     train = pickle.load(f)
    from dotenv import load_dotenv
    import os
    load_dotenv(override=True)
    from huggingface_hub import login
    from datasets import load_dataset
    hf_token = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')
    login(hf_token, add_to_git_credential=True)
    DATASET_NAME = f"alistermarc/llama3-pricer-2025-08-30_02.01.02"
    dataset = load_dataset(DATASET_NAME)
    train = dataset['train']
  
    total_price = 0
    for item in train:
        total_price += item["price"]
    
    return total_price / len(train)

AVERAGE_PRICE = calculate_average_price()

def average_pricer(item):
    return AVERAGE_PRICE

def run():
    Tester.test(average_pricer)