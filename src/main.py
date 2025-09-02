import sys
import os
sys.path.append(os.getcwd())
import argparse
from src.data_curation.curate import curate_data
from src.random_pricer.run import run as random_pricer_run
from src.average_pricer.run import run as average_pricer_run
from src.frontier.run import run as frontier_run
from src.frontier_finetuning.run import run as frontier_finetuning_run
from src.open_source_prediction.run import run as open_source_prediction_run
from src.qlora_finetuning.run import run as qlora_finetuning_run
from src.qlora_finetuned_prediction.run import run as qlora_finetuned_prediction_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["curate_data", "random_pricer", "average_pricer", "frontier", "frontier_finetuning", "open_source_prediction", "open_source_finetuning_prediction", "qlora_finetuning", "qlora_finetuned_prediction"], help="Action to perform")
    parser.add_argument("--model", help="Model to use for the frontier action")
    args = parser.parse_args()

    if args.action == "curate_data":
        curate_data()
    elif args.action == "random_pricer":
        random_pricer_run()
    elif args.action == "average_pricer":
        average_pricer_run()
    elif args.action == "frontier":
        if not args.model:
            raise ValueError("Model must be specified for frontier action")
        frontier_run(args.model)
    elif args.action == "frontier_finetuning":
        frontier_finetuning_run()
    elif args.action == "open_source_prediction":
        open_source_prediction_run()
    elif args.action == "qlora_finetuning":
        qlora_finetuning_run()
    elif args.action == "qlora_finetuned_prediction":
        qlora_finetuned_prediction_run()
