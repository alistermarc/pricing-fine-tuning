import pickle
from src.tester import Tester

def calculate_average_price():
    with open("data/train.pkl", "rb") as f:
        train = pickle.load(f)
    
    total_price = 0
    for item in train:
        total_price += item.price
    
    return total_price / len(train)

AVERAGE_PRICE = calculate_average_price()

def average_pricer(item):
    return AVERAGE_PRICE

def run():
    Tester.test(average_pricer)