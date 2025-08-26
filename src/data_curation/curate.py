import pickle
from sklearn.model_selection import train_test_split
from src.data_curation.loader import ItemLoader
from src.config import DATASET_CATEGORY

def curate_data():
    """
    Loads the data, processes it, and saves it to train.pkl and test.pkl
    """
    loader = ItemLoader(DATASET_CATEGORY)
    items = loader.load()

    train_items, test_items = train_test_split(items, test_size=0.2, random_state=42)

    with open("data/train.pkl", "wb") as f:
        pickle.dump(train_items, f)

    with open("data/test.pkl", "wb") as f:
        pickle.dump(test_items, f)

    print(f"Saved {len(train_items)} training items to data/train.pkl")
    print(f"Saved {len(test_items)} testing items to data/test.pkl")
