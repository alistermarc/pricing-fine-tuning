import pickle
import random
from src.data_curation.loader import ItemLoader
from src.config import DATASET_CATEGORY

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
