import random
from src.tester import Tester

def random_pricer(item):
    return random.randrange(1,1000) # Set the random seed

random.seed(42)

# Run our TestRunner
def run():
    Tester.test(random_pricer)
