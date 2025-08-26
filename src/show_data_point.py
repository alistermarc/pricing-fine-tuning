import pickle

def show_data_point():
    """
    Loads the train.pkl file and prints the first data point.
    """
    try:
        with open("data/train.pkl", "rb") as f:
            items = pickle.load(f)
        print(items[0])
    except FileNotFoundError:
        print("Error: data/train.pkl not found. Please run the data curation process first.")
