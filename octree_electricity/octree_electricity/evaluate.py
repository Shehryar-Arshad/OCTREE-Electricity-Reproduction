"""
evaluate.py - summarise saved runs and compare them with the paper.  [OURS]

Example
    python evaluate.py                       # every run found in runs/
    python evaluate.py --runs octree full_octree
    python evaluate.py --model model         # also: accuracy, precision, recall, F1, confusion matrix of the saved model
"""
import argparse
import os

import pandas as pd

import octree_core as oc

p = argparse.ArgumentParser(description="Summarise OCTree runs")
p.add_argument("--runs", nargs="*", default=None, help="run names (default: all folders in runs/)")
p.add_argument("--out_dir", default="runs")
p.add_argument("--model", default=None, help="folder of a saved model (e.g. model) to score on its test split")
args = p.parse_args()

names = args.runs or sorted(d for d in os.listdir(args.out_dir) if os.path.isdir(os.path.join(args.out_dir, d)))
pd.set_option("display.width", 200)
table = oc.summarise(names, out_dir=args.out_dir)
print(table.to_string(index=False) if len(table) else "No finished runs found.")
print(f"\nPaper (electricity, XGBoost): baseline {oc.PAPER['baseline']}%, OCTree {oc.PAPER['octree']}%, "
      f"OCTree without decision tree reasoning {oc.PAPER['octree_no_tree']}%")

for name in names:
    for seed in (0, 1, 2):
        run = oc.load_run(name, seed, args.out_dir)
        if isinstance(run, dict) and run.get("final"):
            print(f"\n{name} | seed {seed}: {run['final']['baseline_error']:.2f}% -> {run['final']['test_error']:.2f}%")
            for rule in run["final"]["rules"]:
                print("   ", rule)

if args.model:
    # Score the saved model exactly as the app uses it: raw test values -> preprocessing -> model.
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    pipe, clf = oc.load_pipeline(args.model)
    split = oc.load_split(pipe["seed"])
    raw = split["xtest"] * (split["raw_max"] - split["raw_min"]) + split["raw_min"]
    labels, _ = oc.predict(pipe, clf, raw)
    y_pred = (pd.Series(labels) == "UP").astype(int).to_numpy()
    y = split["ytest"].astype(int)
    print(f"\nSaved model in {args.model}/ | split seed {pipe['seed']} | rules: {[f['rule'] for f in pipe['generated_features']] or 'none'}")
    print(f"accuracy  {accuracy_score(y, y_pred):.4f}   (test error {100 * (1 - accuracy_score(y, y_pred)):.2f}%)")
    print(f"precision {precision_score(y, y_pred):.4f}   recall {recall_score(y, y_pred):.4f}   F1 {f1_score(y, y_pred):.4f}   (positive class = UP)")
    cm = confusion_matrix(y, y_pred)
    print("confusion matrix (rows = true DOWN/UP, columns = predicted DOWN/UP):")
    print(cm)
