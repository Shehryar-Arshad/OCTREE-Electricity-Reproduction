"""
predict.py - inference from the command line.  [OURS]

Examples
    python predict.py --example
    python predict.py --values 0.4257 2 0.5106 0.0703 0.4306 0.0047 0.4229 0.4149
    python predict.py --csv my_rows.csv --out predictions.csv

Input : eight numbers per row, in this order and on the dataset's own scale:
        date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer
Output: the label (UP or DOWN) and the model's probability of UP.
"""
import argparse

import numpy as np
import pandas as pd

import octree_core as oc

parser = argparse.ArgumentParser(description="Predict UP/DOWN with the trained OCTree + XGBoost model")
parser.add_argument("--model_dir", default="model")
parser.add_argument("--values", nargs=8, type=float, help="one row: " + " ".join(oc.RAW_COLUMNS))
parser.add_argument("--csv", help="CSV file with the eight columns")
parser.add_argument("--out", help="where to save predictions for --csv")
parser.add_argument("--example", action="store_true", help="predict a saved test-set example")
args = parser.parse_args()

pipe, clf = oc.load_pipeline(args.model_dir)

if args.csv:
    df = pd.read_csv(args.csv)
    labels, p_up = oc.predict(pipe, clf, df[pipe["raw_columns"]].to_numpy(dtype=float))
    df["prediction"], df["probability_UP"] = labels, np.round(p_up, 4)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Saved {len(df)} predictions to {args.out}")
    else:
        print(df.to_string(index=False))
else:
    if args.values:
        row, truth = args.values, None
    else:
        ex = pipe["examples"][0]
        row, truth = ex["values"], ex["true_label"]
    labels, p_up = oc.predict(pipe, clf, [row])
    print("Input     :", dict(zip(pipe["raw_columns"], row)))
    print(f"Prediction: {labels[0]}  (probability of UP = {p_up[0]:.3f})")
    if truth:
        print("True label:", truth)
