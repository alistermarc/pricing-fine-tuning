import pickle
import random
from src.data_curation.loader import ItemLoader
from src.config import DATASET_CATEGORY, CURATED_DATASET_NAME
from datasets import Dataset, DatasetDict

def curate_data():
    """
    Loads the data, processes it, and saves it to train.pkl and test.pkl
    """
    items = []
    for category in DATASET_CATEGORY:
        loader = ItemLoader(category)
        items.extend(loader.load())

    random.seed(42)
    random.shuffle(items)
    train_items = items[:25_000]
    test_items = items[25_000:27_000]

    with open("data/train.pkl", "wb") as f:
        pickle.dump(train_items, f)

    with open("data/test.pkl", "wb") as f:
        pickle.dump(test_items, f)

    print(f"Divided into a training set of {len(train_items):,} items and test set of {len(test_items):,} items")

    # Prepare for Hugging Face
    train_prompts = [item.prompt for item in train_items]
    train_prices = [item.price for item in train_items]
    test_prompts = [item.prompt() for item in test_items]
    test_prices = [item.price for item in test_items]

    train_dataset = Dataset.from_dict({"prompt": train_prompts, "price": train_prices})
    test_dataset = Dataset.from_dict({"prompt": test_prompts, "price": test_prices})

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # Push to hub
    print(f"Pushing dataset to {CURATED_DATASET_NAME}")
    dataset_dict.push_to_hub(CURATED_DATASET_NAME, private=False)
    print("Dataset pushed to Hugging Face Hub.")