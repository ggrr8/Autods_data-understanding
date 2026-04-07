import argparse
import json
from pathlib import Path

import pandas as pd

from data_understanding_agent import (
    AgentConfig,
    DataUnderstandingAgent,
    load_planner_input,
)


def parse_args():
    parser = argparse.ArgumentParser(description="AUTODS Data Understanding Agent")

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--target-column", type=str, default=None)
    parser.add_argument(
        "--problem-type",
        type=str,
        default=None,
        choices=["classification", "regression"],
    )
    parser.add_argument("--output-dir", type=str, default="data_understanding_outputs")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--planner-input", type=str, default=None)

    return parser.parse_args()


def load_dataframe(data_path: str):
    if data_path.endswith(".csv"):
        return pd.read_csv(data_path)
    elif data_path.endswith(".parquet") or data_path.endswith(".pq"):
        return pd.read_parquet(data_path)
    else:
        raise ValueError("Only CSV or Parquet supported")


def main():
    args = parse_args()

    df = load_dataframe(args.data_path)
    dataset_name = args.dataset_name or Path(args.data_path).stem

    config = AgentConfig(
        output_dir=args.output_dir,
        target_column=args.target_column,
        problem_type=args.problem_type,
        dataset_name=dataset_name,
        random_state=args.random_state,
    )

    planner_input = None
    if args.planner_input:
        planner_input = load_planner_input(args.planner_input)

    agent = DataUnderstandingAgent(config, planner_input=planner_input)
    result = agent.run(df)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()