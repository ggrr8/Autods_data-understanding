import argparse
import json

from data_understanding_agent import AgentConfig, DataUnderstandingAgent


def parse_args():
    parser = argparse.ArgumentParser(description="AUTODS Data Understanding Agent")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to raw CSV or Parquet dataset",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Target column name",
    )
    parser.add_argument(
        "--problem-type",
        type=str,
        default=None,
        choices=["classification", "regression"],
        help="Problem type for downstream workflow",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_understanding_outputs",
        help="Directory for exported artifacts",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset display name",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed metadata for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = AgentConfig(
        data_path=args.data_path,
        target_column=args.target_column,
        problem_type=args.problem_type,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        random_state=args.random_state,
    )

    agent = DataUnderstandingAgent(config)
    result = agent.run()

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
