"""
train.py - run OCTree from the command line and save the final model.  [OURS]

Examples
    # OCTree with an LLM (needs a key; see README)
    python train.py --seed 0 --provider gemini --model gemini-3.5-flash-lite --iters 50 --features 2

    # score-only ablation (no decision tree in the prompt)
    python train.py --seed 0 --provider gemini --model gemini-3.5-flash-lite --no_tree

Runs are saved in runs/<run_name>/seed<k>.json and resume automatically if interrupted.
With --save_model the final XGBoost model is written to model/ for predict.py and app.py.
"""
import argparse

import octree_core as oc

p = argparse.ArgumentParser(description="OCTree on the electricity dataset")
p.add_argument("--seed", type=int, default=0, choices=[0, 1, 2])
p.add_argument("--provider", default="gemini", help="gemini | groq | openrouter | ollama | hf_local")
p.add_argument("--model", default="", help="model name at the provider")
p.add_argument("--api_key", default=None, help="or set GROQ_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY")
p.add_argument("--iters", type=int, default=50, help="proposals per generated feature (paper: 50)")
p.add_argument("--features", type=int, default=2, help="maximum number of generated features")
p.add_argument("--no_tree", action="store_true", help="ablation: do not show decision trees to the LLM")
p.add_argument("--trees", type=int, default=None, help="override XGBoost n_estimators (quick tests only)")
p.add_argument("--pause", type=float, default=3.0, help="seconds between LLM calls")
p.add_argument("--run_name", default=None)
p.add_argument("--save_model", action="store_true", help="train and save the final model to model/")
args = p.parse_args()

if args.provider == "hf_local":
    proposer = oc.HFLocalLLM(args.model or "kykim0/Llama-2-7b-ultrachat200k-2e")
else:
    proposer = oc.ChatLLM(args.provider, args.model, api_key=args.api_key, min_seconds_between_calls=args.pause)

name = args.run_name or ("octree_score_only" if args.no_tree else "octree")
run = oc.run_octree(args.seed, proposer, name, n_iters=args.iters, max_features=args.features,
                    use_tree=not args.no_tree, n_estimators=args.trees)

print("\nRules kept:", run["final"]["rules"] or "none")
print(f"Test error: {run['final']['test_error']:.2f}%  (baseline {run['final']['baseline_error']:.2f}%, "
      f"paper: {oc.PAPER['baseline']}% -> {oc.PAPER['octree']}%)")

if args.save_model:
    pipe = oc.train_final_model(args.seed, run["final"]["rules"], n_estimators=args.trees,
                                extra={"run": name, "proposer": run["config"]["proposer"]})
    print(f"Saved model/ (test error {pipe['metrics']['test_error_percent']:.2f}%)")
