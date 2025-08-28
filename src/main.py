import sys
import os
sys.path.append(os.getcwd())
import argparse
from src.data_curation.curate import curate_data
from src.show_data_point import show_data_point
from src.random_pricer.run import run as random_pricer_run
from src.average_pricer.run import run as average_pricer_run
from src.frontier.run import run as frontier_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["curate_data", "show_data_point", "random_pricer", "average_pricer", "frontier"], help="Action to perform")
    parser.add_argument("--model", help="Model to use for the frontier action")
    args = parser.parse_args()

    if args.action == "curate_data":
        curate_data()
    elif args.action == "show_data_point":
        show_data_point()
    elif args.action == "random_pricer":
        random_pricer_run()
    elif args.action == "average_pricer":
        average_pricer_run()
    elif args.action == "frontier":
        if not args.model:
            raise ValueError("Model must be specified for frontier action")
        frontier_run(args.model)
