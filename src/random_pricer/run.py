import random
from src.tester import Tester

def random_pricer(item):
    return random.randrange(1,1000)

random.seed(42)

def run():
    Tester.test(random_pricer)
