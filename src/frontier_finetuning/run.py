import os
import re
import math
import json
import random
import time
from dotenv import load_dotenv
from huggingface_hub import login
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import Counter
from openai import OpenAI
from anthropic import Anthropic

from src.data_curation.item import Item
from src.tester import Tester

def run():
    # environment
    load_dotenv(override=True)
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
    os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')

    # Log in to HuggingFace
    hf_token = os.environ['HF_TOKEN']
    login(hf_token, add_to_git_credential=True)

    openai = OpenAI()

    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Let's avoid curating all our data again! Load in the pickle files:
    with open(os.path.join(project_root, 'data', 'train.pkl'), 'rb') as file:
        train = pickle.load(file)

    with open(os.path.join(project_root, 'data', 'test.pkl'), 'rb') as file:
        test = pickle.load(file)

    # OpenAI recommends fine-tuning with populations of 50-100 examples
    # But as our examples are very small, I'm suggesting we go with 200 examples (and 1 epoch)
    fine_tune_train = train[:200]
    fine_tune_validation = train[200:250]

    # Step 1
    # Prepare our data for fine-tuning in JSONL (JSON Lines) format and upload to OpenAI
    def messages_for(item):
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": f"Price is ${item.price:.2f}"}
        ]

    def make_jsonl(items):
        result = ""
        for item in items:
            messages = messages_for(item)
            messages_str = json.dumps(messages)
            result += '{"messages": ' + messages_str +'}\n'
        return result.strip()

    def write_jsonl(items, filename):
        with open(filename, "w") as f:
            jsonl = make_jsonl(items)
            f.write(jsonl)

    train_jsonl_path = os.path.join(project_root, "fine_tune_train.jsonl")
    validation_jsonl_path = os.path.join(project_root, "fine_tune_validation.jsonl")

    write_jsonl(fine_tune_train, train_jsonl_path)
    write_jsonl(fine_tune_validation, validation_jsonl_path)

    with open(train_jsonl_path, "rb") as f:
        train_file = openai.files.create(file=f, purpose="fine-tune")

    with open(validation_jsonl_path, "rb") as f:
        validation_file = openai.files.create(file=f, purpose="fine-tune")

    # Step 2
    # And now time to Fine-tune!
    wandb_integration = {"type": "wandb", "wandb": {"project": "gpt-pricer"}}

    job = openai.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=validation_file.id,
        model="gpt-4o-mini-2025-08-29",
        seed=42,
        hyperparameters={"n_epochs": 1},
        integrations = [wandb_integration],
        suffix="pricer"
    )

    job_id = job.id
    print(f"Fine-tuning job started with ID: {job_id}")

    # Wait for the fine-tuning job to complete
    while True:
        job_status = openai.fine_tuning.jobs.retrieve(job_id).status
        print(f"Fine-tuning job status: {job_status}")
        if job_status == 'succeeded':
            break
        elif job_status == 'failed':
            print("Fine-tuning job failed.")
            return
        time.sleep(60)

    # Step 3
    # Test our fine tuned model
    fine_tuned_model_name = openai.fine_tuning.jobs.retrieve(job_id).fine_tuned_model

    def messages_for_test(item):
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]

    def get_price(s):
        s = s.replace('$','').replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0

    def gpt_fine_tuned(item):
        response = openai.chat.completions.create(
            model=fine_tuned_model_name,
            messages=messages_for_test(item),
            seed=42,
            max_tokens=7
        )
        reply = response.choices[0].message.content
        return get_price(reply)

    Tester.test(gpt_fine_tuned)