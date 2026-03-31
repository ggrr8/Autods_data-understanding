import argparse
import json
import pandas as pd

from data_understanding_agent import AgentConfig, DataUnderstandingAgent


def parse_args():
    parser = argparse.ArgumentParser(description="AUTODS Data Understanding Agent")

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--target-column", type=str, default=None)
    parser.add_argument("--problem-type", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="data_understanding_outputs")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


def load_dataframe(data_path: str):
    if data_path.endswith(".csv"):
        return pd.read_csv(data_path)
    elif data_path.endswith(".parquet"):
        return pd.read_parquet(data_path)
    else:
        raise ValueError("Only CSV or Parquet supported")


def main():
    args = parse_args()

    df = load_dataframe(args.data_path)

    config = AgentConfig(
        output_dir=args.output_dir,
        target_column=args.target_column,
        problem_type=args.problem_type,
        dataset_name=args.dataset_name,
        random_state=args.random_state,
    )

    agent = DataUnderstandingAgent(config)

    result = agent.run(df)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
