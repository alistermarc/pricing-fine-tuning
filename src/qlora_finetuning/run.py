import os
import torch
import wandb
from datetime import datetime
from datasets import Dataset, load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import pickle
from src.data_curation.item import Item
from dotenv import load_dotenv
from src.config import BASE_MODEL, HF_USER, PROJECT_NAME

load_dotenv(override=True)
hf_token = os.getenv('HF_TOKEN')
wandb_api_key = os.getenv('WANDB_API_KEY')

def run():
    
    login(hf_token, add_to_git_credential=True)
    wandb.login(key=wandb_api_key)

    # Hugging Face Configuration
    RUN_NAME =  f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
    PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
    HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

    with open('data/train.pkl', 'rb') as file:
        train_items = pickle.load(file)

    with open('data/test.pkl', 'rb') as file:
        test_items = pickle.load(file)

    train_prompts = [item.prompt for item in train_items]
    train_prices = [item.price for item in train_items]
    test_prompts = [item.prompt for item in test_items]
    test_prices = [item.price for item in test_items]

    train = Dataset.from_dict({"prompt": train_prompts, "price": train_prices})
    test = Dataset.from_dict({"prompt": test_prompts, "price": test_prices})

    split_ratio = 0.1
    TRAIN_SIZE = 200

    train = train.select(range(TRAIN_SIZE))
    total_size = len(train)
    val_size = int(total_size * split_ratio)

    val_data = train.select(range(val_size))
    train_data = train.select(range(val_size, total_size))

    print(f"Train data size     : {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Test data size      : {len(test)}")

    # wandb Configuration
    LOG_TO_WANDB = True
    os.environ["WANDB_PROJECT"] = PROJECT_NAME
    os.environ["WANDB_LOG_MODEL"] = "checkpoint" if LOG_TO_WANDB else "end"
    os.environ["WANDB_WATCH"] = "gradients"
    if LOG_TO_WANDB:
      wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    # Load the Tokenizer and Model
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

    print(f"Memory footprint: {base_model.get_memory_footprint() / 1e6:.1f} MB")

    # Prepare the Data with a Data Collator
    response_template = "Price is $"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # Define the QLoRA Configuration (LoraConfig)
    LORA_R = 32
    LORA_ALPHA = 64
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    LORA_DROPOUT = 0.1

    lora_parameters = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set the Training Parameters (SFTConfig)
    EPOCHS = 1
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2
    MAX_SEQUENCE_LENGTH = 182
    LEARNING_RATE = 1e-4
    LR_SCHEDULER_TYPE = 'cosine'
    WARMUP_RATIO = 0.03
    OPTIMIZER = "paged_adamw_32bit"
    SAVE_STEPS = 200
    STEPS = 20
    save_total_limit = 10

    train_parameters = SFTConfig(
        output_dir=PROJECT_RUN_NAME,
        run_name=RUN_NAME,
        dataset_text_field="text",
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_steps=-1,
        group_by_length=True,
        eval_strategy="steps",
        eval_steps=STEPS,
        per_device_eval_batch_size=1,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        optim=OPTIMIZER,
        weight_decay=0.001,
        max_grad_norm=0.3,
        fp16=False,
        bf16=True,
        logging_steps=STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=save_total_limit,
        report_to="wandb" if LOG_TO_WANDB else None,
        push_to_hub=True,
        hub_strategy="end",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Initialize the Fine-Tuning Trainer (SFTTrainer)
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_parameters,
        args=train_parameters,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Run Fine-Tuning and Push to Hub
    fine_tuning.train()
    print(f"✅ Best model pushed to HF Hub: {HUB_MODEL_NAME}")

    if LOG_TO_WANDB:
      wandb.finish()