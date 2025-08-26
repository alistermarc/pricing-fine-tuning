import sys
import os
sys.path.append(os.getcwd())
import argparse
from src.data_curation.curate import curate_data
from src.show_data_point import show_data_point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["curate_data", "show_data_point"], help="Action to perform")
    args = parser.parse_args()

    if args.action == "curate_data":
        curate_data()
    elif args.action == "show_data_point":
        show_data_point()
