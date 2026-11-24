# OCTree on the Electricity Dataset

Small-scale reproduction of **"Optimized Feature Generation for Tabular Data via LLMs with Decision Tree
Reasoning"** (Nam et al., NeurIPS 2024) on the electricity dataset (OpenML 44156) in the context-agnostic
setting, with our own experiments and a deployed demo.

Machine Learning (101700), IBA. Team: Muhammad Salman Malik, Muhammad Shehryar Arshad, Manahil Imran.

## Links

- Final report (PDF): `Final_Report.pdf` in this repository
- Code repository: <add your GitHub link here>
- Google Drive copy (optional): <add the link here, or delete this line>
- Trained model: `model/` folder in this repository (`xgb_model.json` + `pipeline.json`)

## What the method does

XGBoost is trained on eight anonymous columns `x1 … x8`. An LLM proposes a rule that creates a new column,
such as `x9 = [x4 * x1]`. The column is added, XGBoost is retrained, and the validation accuracy becomes the
rule's score. A small decision tree (CART) fitted on the new data is written out as text. Rule, score and tree
are fed back to the LLM, which proposes a better rule. After 50 proposals the best rule on validation data is
kept, and the process repeats for another column until validation stops improving.

## Files

| File | Purpose |
|---|---|
| `octree_core.py` | All method code: data, XGBoost evaluation, rule parsing, CART-to-text, prompt, LLM backends, optimisation loop, final model, inference |
| `Deliverable_2_Electricity_EDA_and_Preprocessing.ipynb` | EDA and preprocessing (creates `dataset/`) |
| `OCTree_Reproduction_and_Experiments.ipynb` | Runs everything step by step: baseline, OCTree, comparison with the paper, our experiments, final model |
| `train.py` | **Training code**: run OCTree (or the no-tree ablation) from the command line |
| `evaluate.py` | **Evaluation code**: summarise saved runs and compare with the paper |
| `predict.py` | **Inference code**: predictions from the command line |
| `app.py` | **Deployment code**: Streamlit web app |
| `requirements.txt` | Dependencies |
| `dataset/` | Created by the preprocessing notebook (or automatically on first run) |
| `runs/` | Saved optimisation runs (one JSON per run and split) |
| `model/` | **Trained model**: `xgb_model.json` (XGBoost weights) and `pipeline.json` (scaling values and generated-feature rules) |
| `screenshots/` | Screenshots of the running app |
| `sample_input.csv` | Ten real test rows to try the CSV upload |

## Setup

```
pip install -r requirements.txt
```

Python 3.10 or newer. A GPU is not required: XGBoost uses one if present, and the LLM is called through an API.

### LLM access

The method needs an LLM that returns text. We call one through an OpenAI-compatible API, so any of these
works; each offers a free key at the time of writing (check current limits on the provider's site):

| Provider (`--provider`) | Key variable | Notes |
|---|---|---|
| `groq` | `GROQ_API_KEY` | Fast, no card needed |
| `gemini` | `GEMINI_API_KEY` | Google AI Studio key |
| `openrouter` | `OPENROUTER_API_KEY` | Models whose name ends in `:free` |
| `ollama` | none | A model running on your own computer |
| `hf_local` | none | The authors' fine-tuned Llama 2 run on a GPU, as in the original code |

Set the key as an environment variable, or paste it in the notebook's settings cell.

**Our results** were produced with `gemini` / `gemini-3.5-flash-lite` (Google AI Studio free tier). Its free
tier allowed 500 requests per day at the time of our runs; one full OCTree run (3 splits, 50 proposals,
2 features) needed about 340 calls. Groq's free daily token limit is too small for a full run in one day.

## How to run

**Notebook (recommended).** Open `OCTree_Reproduction_and_Experiments.ipynb`, fill in the settings cell, and
run the cells in order. Start with `PROFILE = "quick"`, then switch to `"full"`.

**Command line.**

```
python train.py --seed 0 --provider gemini --model gemini-3.5-flash-lite --save_model   # OCTree + final model
python train.py --seed 0 --provider gemini --model gemini-3.5-flash-lite --no_tree      # ablation: scores only
python evaluate.py                                                           # results table
python evaluate.py --model model                                             # accuracy, precision, recall, F1, confusion matrix of the saved model
```

Runs are saved after every step in `runs/`. If the LLM reaches a rate or daily limit, run the same command
again and it continues from where it stopped.

## Deployment

Pipeline: **Input → Preprocessing → Trained model → Prediction → Output**

**Start the app** (after a model has been saved to `model/`):

```
streamlit run app.py
```

It opens at `http://localhost:8501`.

**Dependencies:** `streamlit`, `xgboost`, `numpy`, `pandas`, `scikit-learn` (all in `requirements.txt`).
No LLM or internet access is needed at prediction time: the generated-feature rules are stored in
`model/pipeline.json`.

**Input format:** eight numbers on the dataset's own (normalised) scale, in this order:

| Column | Meaning | Range |
|---|---|---|
| `date` | Date | 0 – 1 |
| `day` | Day-of-week code | 0 – 6 |
| `period` | Half-hour of the day (slot ÷ 47) | 0 – 1 |
| `nswprice`, `nswdemand` | New South Wales price and demand | 0 – 1 |
| `vicprice`, `vicdemand` | Victoria price and demand | 0 – 1 |
| `transfer` | Scheduled transfer between states | 0 – 1 |

**Preprocessing (done by the app):** each value is min-max scaled with the training data's minimum and
maximum; the generated features are computed from the scaled values with the saved rules and scaled the same way.

**Output format:** the label `UP` or `DOWN` (NSW price relative to its 24-hour moving average) and the
model's probability of `UP`. The CSV tab returns the uploaded file with `prediction` and `probability_UP` columns.

**Example inference:**

```
python predict.py --example
python predict.py --values 0.4257 2 0.5106 0.0703 0.4306 0.0047 0.4229 0.4149
```

## What came from where

Original code: https://github.com/jaehyun513/OCTree (`ours/train_xg.py`, `ours/utils_xg.py`, `ours/params.py`).
Each function in `octree_core.py` carries one of these labels.

| Component | Status | Detail |
|---|---|---|
| Prompt text (`build_prompt`) | **Reused** | Authors' `gen_prompt`, wording unchanged. We added the option to leave out the decision trees. |
| Tree written as text (`tree_to_text`) | **Reused** | Authors' `tree_to_code`. |
| Tuned XGBoost hyperparameters (`XGB_PARAMS`) | **Reused** | Authors' published values for electricity, splits 0–2. |
| Rule-matching regular expression | **Reused** | With added handling of thinking blocks and answers without brackets. |
| XGBoost evaluation, CART selection, initial rule, optimisation loop | **Adapted** | Same logic as `evaluate`, `get_cart` and `train_xg.py`, rewritten. |
| Local Llama 2 backend (`HFLocalLLM`) | **Adapted** | Authors' `load_model` and `use_api`. |
| Data preparation | **Ours** | From our preprocessing notebook; same split and scaling as the authors' `preprocess.ipynb`. |
| Safe rule evaluation (`compile_rule`, `apply_rule`) | **Ours** | Replaces `exec()` of LLM-written code. |
| API LLM backend (`ChatLLM`), rate-limit handling | **Ours** | Replaces the local model. |
| Hand-made features (`MANUAL_FEATURES`, `run_manual_features`) and data-size experiment (`run_data_size`) | **Ours** | Our experiments (notebook Section 7). |
| Timing and token logging | **Ours** | Cost figures. |
| Saving and resuming runs, multi-feature bookkeeping, summaries | **Ours** | |
| Final model, inference, `predict.py`, `app.py`, `train.py`, `evaluate.py` | **Ours** | |

### Changes made to make the method feasible

| Paper / original code | This project | Why |
|---|---|---|
| 19 datasets without descriptions, 6 with | Electricity only | Course scope |
| Fine-tuned Llama 2 (7B) on a local GPU | An LLM reached through a free API (selectable) | No GPU needed |
| 400 Optuna trials per split to tune XGBoost | Authors' published tuned values reused | Saves the tuning cost |
| LLM output run with `exec()` | Checked and evaluated with a restricted set of maths functions | Safety |
| CART depths 1–3 in the code, "maximum depth 4" in the paper | Depths 1–4 (set `cart_max_depth=3` for the code's behaviour) | Follows the paper's text |
| A repeated rule is evaluated again | A repeated rule is rejected and a new one requested | The paper asks for rules not already tried |
| A rule giving inf/NaN is recorded, then retried | It is rejected before being recorded | Keeps the history clean |
| Features added until validation stops improving | Same, capped at `--features` (default 2) | LLM call budget |

## Results

Paper, electricity, XGBoost test error: baseline **8.32%**, OCTree **6.65%**, OCTree without decision tree
reasoning **6.89%**.

Our results (Gemini 3.5 Flash-Lite, 50 proposals per feature, at most 2 features, 3 splits; test error %,
mean ± std over the splits; relative reduction from the baseline in brackets):

| Run | Test error % |
|---|---|
| XGBoost, original columns (baseline) | 8.50 ± 0.16 |
| OCTree | 7.66 ± 0.10 (9.9%) |
| OCTree without decision tree reasoning | 7.71 ± 0.29 (9.4%) |
| Hand-made features, same selection rule (Experiment 1) | 8.44 ± 0.07 (0.8%) |
| Deployed model (split 0) | 7.76 (precision 91.9%, recall 92.5%, F1 92.2% for UP) |

The notebook (Sections 6 to 8) and `python evaluate.py --model model` reproduce these numbers from the
saved runs in `runs/`.

## Our experiments (notebook Section 7)

1. **Baseline comparison: hand-made features vs OCTree.** Seven features designed from what the columns mean
   (e.g. NSW price minus Victoria price, time of day as a circle, peak hours), chosen with exactly OCTree's
   selection rule. Question: does an LLM that cannot see the column names beat features designed by a person
   who can?
2. **Data size.** XGBoost trained on 10%, 25%, 50% and 100% of the training rows, with the original columns,
   with the hand-made features and with OCTree's features. Question: do generated features help more when
   training data is scarce?

Neither needs extra LLM calls.

## References

- Nam, J., Kim, K., Oh, S., Tack, J., Kim, J., & Shin, J. (2024). Optimized feature generation for tabular
  data via LLMs with decision tree reasoning. *NeurIPS 37*.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning
  on typical tabular data? *NeurIPS 35*.
- Official OCTree code: https://github.com/jaehyun513/OCTree
- Dataset: https://www.openml.org/d/44156
