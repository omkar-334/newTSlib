import argparse
import json
import os

import pandas as pd


def parse_json(data: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.reset_index()
    df = df.rename(columns={"index": "experiment_name"})
    return df


def parse_stf_json(data: dict) -> pd.DataFrame:
    records = []
    for exp_name, info in data.items():
        record = {
            "experiment_name": exp_name,
            "time": info["time"],
        }
        for metric, periods in info.items():
            if metric == "time":
                continue
            for period, val in periods.items():
                record[f"{metric}_{period.lower()}"] = val
        records.append(record)
    df = pd.DataFrame(records)

    def remove_time_period(exp_name: str) -> str:
        parts = exp_name.split("_")
        if len(parts) > 5:
            del parts[4]  # remove time period
        return "_".join(parts[:5])

    df["experiment_name"] = df["experiment_name"].apply(remove_time_period)

    # Drop duplicates based on base_experiment_name, keep first
    df_unique = df.drop_duplicates(
        subset=["experiment_name"], keep="first"
    ).reset_index(drop=True)
    return df_unique


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a JSON file to csv.")
    parser.add_argument("json_path", type=str, help="Relative path to the JSON file")
    json_path = parser.parse_args().json_path

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    output_path = os.path.splitext(json_path)[0]

    if "STF" in json_path:
        df = parse_stf_json(data)
        output_path = os.path.abspath(output_path + ".xlsx")
        df.to_excel(output_path, index=False)
    else:
        df = parse_json(data)
        output_path = os.path.abspath(output_path + ".csv")
        df.to_csv(output_path, index=False)
