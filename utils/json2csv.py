import argparse
import json
import os

import pandas as pd


def convert_json_to_csv(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.reset_index()
    df = df.rename(columns={"index": "experiment_name"})

    csv_path = os.path.splitext(json_path)[0] + ".csv"
    csv_path = os.path.abspath(csv_path)

    df.to_csv(csv_path, index=False)

    print(f"csv file saved at: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a JSON file to csv.")
    parser.add_argument("json_path", type=str, help="Relative path to the JSON file")
    args = parser.parse_args()

    convert_json_to_csv(args.json_path)
