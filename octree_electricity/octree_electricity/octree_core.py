"""
octree_core.py
==============
Small-scale reproduction of OCTree (Nam et al., NeurIPS 2024) on the electricity
dataset (OpenML 44156) in the context-agnostic setting.

Where the code comes from (see README.md for the full table):
  [REUSED]   copied from the authors' repository (github.com/jaehyun513/OCTree) with
             only cosmetic edits: the prompt text, the tree-to-text function and the
             tuned XGBoost hyperparameters.
  [ADAPTED]  the authors' logic, rewritten by us: evaluation, CART selection, the
             initial rule, the optimisation loop and the local Hugging Face LLM loader.
  [OURS]     written by us: data preparation, safe rule evaluation, API-based LLM
             backends, our two experiments (hand-made features, data size),
             checkpointing and resuming,
             multi-feature bookkeeping, the final model, inference and the summaries.

The file is organised in the order the method runs:
  1. data            4. decision tree reasoning     7. the optimisation loop
  2. XGBoost         5. prompt                      8. final model and inference
  3. rules           6. proposers (LLM / random)    9. summaries
"""

import ast
import json
import math
import os
import random
import re
import time
from types import SimpleNamespace

import numpy as np
from sklearn import tree as sktree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import _tree
from xgboost import XGBClassifier

try:                                   # hide 'mismatched devices' warnings printed on every prediction
    import xgboost as _xgb
    _xgb.set_config(verbosity=0)
except Exception:
    pass

# ----------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------
OPENML_ID = 44156
RAW_COLUMNS = ["date", "day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
TARGET_COL = "class"
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2

# Test error rates (%) reported in the paper for electricity (Tables 2 and 6).
PAPER = {"baseline": 8.32, "octree": 6.65, "octree_no_tree": 6.89}


def seed_everything(seed: int = 0) -> None:
    """Fix Python and NumPy randomness."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==================================================================================
# 1. DATA  [OURS] - same steps as our preprocessing notebook (Deliverable 2)
# ==================================================================================
def _load_raw(data_dir: str):
    """Return the raw table as a DataFrame, downloading it once and caching it as CSV."""
    import pandas as pd
    cache = os.path.join(data_dir, "electricity_raw.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"day": str})
    from sklearn.datasets import fetch_openml
    df = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto").frame.copy()
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def prepare_data(data_dir: str = "dataset", seeds=(0, 1, 2)) -> None:
    """Create dataset/seed{k}/ (60/20/20 split, min-max scaled) if it does not exist yet.

    Identical to the preprocessing notebook: same permutation, scaler fitted on the
    training part only, same file names as the authors.
    """
    need = [s for s in seeds
            if not (os.path.exists(os.path.join(data_dir, f"seed{s}", "xtrain0.npy"))
                    and os.path.exists(os.path.join(data_dir, f"seed{s}", "metadata.json")))]
    if not need:
        return
    df = _load_raw(data_dir).dropna().reset_index(drop=True)
    df.columns = df.columns.str.strip().str.lower()
    X = df[RAW_COLUMNS].copy()
    X["day"] = X["day"].astype(str).str.strip().astype(int)          # ordinal encoding 0..6
    X = X.to_numpy(dtype=np.float64)
    y = df[TARGET_COL].astype(str).str.strip().str.upper().map({"DOWN": 0, "UP": 1}).to_numpy(dtype=np.int8)
    for seed in need:
        seed_everything(seed)
        perm = np.random.permutation(len(X))
        a, b = int(len(X) * TRAIN_FRAC), int(len(X) * (TRAIN_FRAC + VAL_FRAC))
        parts = {"train": perm[:a], "val": perm[a:b], "test": perm[b:]}
        scaler = MinMaxScaler().fit(X[parts["train"]])
        d = os.path.join(data_dir, f"seed{seed}")
        os.makedirs(d, exist_ok=True)
        for name, idx in parts.items():
            np.save(os.path.join(d, f"x{name}0.npy"), scaler.transform(X[idx]))
            np.save(os.path.join(d, f"y{name}.npy"), y[idx])
        meta = {"openml_id": OPENML_ID, "seed": seed, "split": "random 60/20/20",
                "n_train": int(a), "n_val": int(b - a), "n_test": int(len(X) - b),
                "anonymous_columns": [f"x{i + 1}" for i in range(X.shape[1])],
                "original_columns_private": RAW_COLUMNS,
                "train_min_before_scaling": scaler.data_min_.tolist(),
                "train_max_before_scaling": scaler.data_max_.tolist()}
        with open(os.path.join(d, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)


def load_split(seed: int, data_dir: str = "dataset") -> dict:
    """Load one prepared split. Features are already min-max scaled and anonymous (x1..x8)."""
    prepare_data(data_dir, seeds=(seed,))
    d = os.path.join(data_dir, f"seed{seed}")
    out = {}
    for part in ("train", "val", "test"):
        out["x" + part] = np.load(os.path.join(d, f"x{part}0.npy"), allow_pickle=True).astype(np.float64)
        out["y" + part] = np.load(os.path.join(d, f"y{part}.npy"), allow_pickle=True).astype(np.int8)
    with open(os.path.join(d, "metadata.json")) as f:
        meta = json.load(f)
    out["raw_min"] = np.array(meta["train_min_before_scaling"], dtype=float)   # needed at inference time
    out["raw_max"] = np.array(meta["train_max_before_scaling"], dtype=float)
    out["seed"] = seed
    return out


# ==================================================================================
# 2. XGBOOST
# ==================================================================================
# [REUSED] Tuned hyperparameters published by the authors (ours/params.py). They found
# these with 400 Optuna trials per split; reusing them removes that cost for us.
XGB_PARAMS = {
    0: {'n_estimators': 2500, 'max_depth': 11, 'min_child_weight': 1, 'subsample': 0.9726507667395398,
        'learning_rate': 0.16321391231970242, 'colsample_bylevel': 0.727081198453776,
        'colsample_bytree': 0.66335044088413, 'gamma': 1.1445760172126823e-06,
        'reg_alpha': 0.013952340445639796, 'reg_lambda': 1.046918533765224},
    1: {'n_estimators': 2500, 'max_depth': 11, 'min_child_weight': 1, 'subsample': 0.9726507667395398,
        'learning_rate': 0.16321391231970242, 'colsample_bylevel': 0.727081198453776,
        'colsample_bytree': 0.66335044088413, 'gamma': 1.1445760172126823e-06,
        'reg_alpha': 0.013952340445639796, 'reg_lambda': 1.046918533765224},
    2: {'n_estimators': 5500, 'max_depth': 10, 'min_child_weight': 1, 'subsample': 0.784209521433108,
        'learning_rate': 0.008942234364485453, 'colsample_bylevel': 0.9807875721500408,
        'colsample_bytree': 0.5116183133544309, 'gamma': 0.0004984997256564467,
        'reg_alpha': 1.1163840406145217e-06, 'reg_lambda': 1.0772296260336363},
}

_DEVICE = None


def xgb_device() -> str:
    """'cuda' if an NVIDIA GPU is available here, else 'cpu'. [OURS] (authors hard-coded cuda)."""
    global _DEVICE
    if _DEVICE is None:
        import shutil
        import subprocess
        _DEVICE = "cpu"
        if shutil.which("nvidia-smi"):
            try:
                if subprocess.run(["nvidia-smi"], capture_output=True, timeout=20).returncode == 0:
                    _DEVICE = "cuda"
            except Exception:
                pass
    return _DEVICE


def make_xgb(seed: int, n_estimators=None) -> XGBClassifier:
    """XGBoost with the authors' tuned settings for this split.

    n_estimators can be lowered for quick trial runs; leave it as None for real results.
    """
    params = dict(XGB_PARAMS[seed])
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    return XGBClassifier(**params, tree_method="hist", device=xgb_device(),
                         random_state=0, seed=0, n_jobs=-1)


def fit_and_score(Xtr, ytr, Xva, yva, Xte, yte, seed, n_estimators=None):
    """[ADAPTED from evaluate()] Train XGBoost and return (train, val, test) accuracy and the model."""
    clf = make_xgb(seed, n_estimators)
    clf.fit(Xtr, ytr)
    return (accuracy_score(ytr, clf.predict(Xtr)),
            accuracy_score(yva, clf.predict(Xva)),
            accuracy_score(yte, clf.predict(Xte)), clf)


def scale_column(col_tr, col_va, col_te):
    """Min-max scale a generated column using the training part only (as the authors do)."""
    lo, hi = float(np.min(col_tr)), float(np.max(col_tr))
    span = (hi - lo) if hi > lo else 1.0
    f = lambda c: (np.asarray(c, dtype=float) - lo) / span
    return f(col_tr), f(col_va), f(col_te), lo, hi


# ==================================================================================
# 3. RULES  [OURS] - parse the LLM's answer and evaluate it safely
# ==================================================================================
# The authors run the LLM's text with exec(), which would execute anything the model
# writes. We instead (1) check the expression only contains arithmetic, comparisons and
# calls to a fixed list of maths functions, and (2) evaluate it with no Python built-ins.
# Like the authors, the rule is evaluated one row at a time, so it behaves the same way.
_NP_OK = ["sin", "cos", "tan", "tanh", "sinh", "cosh", "arcsinh", "arctan", "arctan2", "arcsin", "arccos",
          "exp", "expm1", "log", "log1p", "log2", "log10", "sqrt", "cbrt", "abs", "absolute", "fabs",
          "square", "power", "maximum", "minimum", "fmax", "fmin", "where", "clip", "sign", "floor",
          "ceil", "round", "rint", "mod", "fmod", "hypot", "add", "subtract", "multiply", "divide",
          "negative", "reciprocal", "logical_and", "logical_or", "logical_not", "greater", "less",
          "greater_equal", "less_equal", "equal", "heaviside", "sinc", "pi", "e", "mean", "sum",
          "prod", "max", "min", "median", "std", "var"]
_SAFE_NP = SimpleNamespace(**{n: getattr(np, n) for n in _NP_OK if hasattr(np, n)})
_MATH_OK = ["sin", "cos", "tan", "tanh", "sinh", "cosh", "exp", "expm1", "log", "log1p", "sqrt",
            "fabs", "floor", "ceil", "pow", "pi", "e", "atan", "atan2", "hypot"]
_SAFE_MATH = SimpleNamespace(**{n: getattr(math, n) for n in _MATH_OK})
_SAFE_FUNCS = {"abs": abs, "max": max, "min": min, "pow": pow, "round": round,
               "float": float, "int": int, "sum": sum, "np": _SAFE_NP, "numpy": _SAFE_NP, "math": _SAFE_MATH}
_OK_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call,
             ast.Name, ast.Load, ast.Constant, ast.Attribute, ast.Tuple, ast.List, ast.keyword,
             ast.operator, ast.unaryop, ast.boolop, ast.cmpop)


class InvalidRule(ValueError):
    """Raised when an answer cannot be turned into a usable column."""


def extract_rule(text: str, target_idx: int):
    """Find 'x9 = [ ... ]' in the LLM's answer and return it in that exact form, or None.

    [REUSED] the authors' regular expression.  [OURS] thinking blocks are removed first,
    the last match is used, and 'x9 = ...' without brackets is accepted as a fallback.
    """
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE).replace("`", " ")
    hits = re.findall(r"(?<![A-Za-z0-9_])x{}\s*=\s*\[.*?\]".format(target_idx), text, re.DOTALL)
    if hits:
        body = hits[-1]
        expr = body[body.index("[") + 1: body.rindex("]")]
    else:
        m = re.findall(r"x{}\s*=\s*([^\n=]+)".format(target_idx), text)
        if m:
            expr = m[-1]
        else:                                       # [OURS] answer is only "[ ... ]" without "x9 ="
            lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
            last = lines[-1] if lines else ""
            if not (last.startswith("[") and last.endswith("]")):
                return None
            expr = last[1:-1]
    expr = " ".join(expr.split()).strip().rstrip(".;,")
    return f"x{target_idx} = [{expr}]" if expr else None


def rule_expression(rule: str) -> str:
    """'x9 = [x4 * x1]'  ->  'x4 * x1'."""
    return rule[rule.index("[") + 1: rule.rindex("]")].strip()


def compile_rule(rule: str, n_vars: int):
    """Check a rule and turn it into a function f(x1, ..., xn) -> number."""
    expr = rule_expression(rule)
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise InvalidRule(f"not valid Python: {e.msg}")
    names = {f"x{i + 1}" for i in range(n_vars)}
    for node in ast.walk(parsed):
        if not isinstance(node, _OK_NODES):
            raise InvalidRule(f"'{type(node).__name__}' is not allowed in a rule")
        if isinstance(node, ast.Name) and node.id not in names and node.id not in _SAFE_FUNCS:
            raise InvalidRule(f"unknown name '{node.id}'")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy", "math")) \
                    or node.attr.startswith("_"):
                raise InvalidRule("only np.<function> and math.<function> are allowed")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool)):
            raise InvalidRule("only numbers are allowed as constants")
    args = ", ".join(f"x{i + 1}" for i in range(n_vars))
    return eval(f"lambda {args}: ({expr})", {"__builtins__": {}, **_SAFE_FUNCS})


def apply_rule(rule: str, X: np.ndarray) -> np.ndarray:
    """Compute the new column for every row of X. Raises InvalidRule if anything goes wrong."""
    f = compile_rule(rule, X.shape[1])
    try:
        with np.errstate(all="ignore"):
            col = np.array([f(*row) for row in X.tolist()], dtype=float)
    except InvalidRule:
        raise
    except Exception as e:                      # division by zero, wrong arguments, missing function ...
        raise InvalidRule(f"{type(e).__name__}: {e}")
    if col.shape != (X.shape[0],):
        raise InvalidRule("the rule must return one number per row")
    if not np.all(np.isfinite(col)):
        raise InvalidRule("the rule produced inf or NaN")       # the authors reject these too
    return col


# ==================================================================================
# 4. DECISION TREE REASONING
# ==================================================================================
def fit_cart(Xtr, ytr, Xva, yva, seed: int, max_depth: int = 4):
    """[ADAPTED from get_cart()] Fit small CART trees and keep the one with the best validation accuracy.

    The paper says 'maximum depth of 4'; the released code tries depths 1-3. We follow the
    paper by default; pass max_depth=3 to match the released code.
    """
    best, best_val = None, -1.0
    for depth in range(1, max_depth + 1):
        cart = sktree.DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(Xtr, ytr)
        val = accuracy_score(yva, cart.predict(Xva))
        if val > best_val:
            best, best_val = cart, val
    return best


def tree_to_text(cart, feature_names) -> str:
    """[REUSED: tree_to_code()] Write a fitted tree as nested if/else text for the prompt."""
    tree_ = cart.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

    def recurse(node, depth):
        indent = "  " * depth
        result = ""
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            result += f"{indent}if {name} > {threshold:.2f}:\n"
            result += recurse(tree_.children_right[node], depth + 1)
            result += f"{indent}else:\n"
            result += recurse(tree_.children_left[node], depth + 1)
        else:
            if tree_.value[node][0][0] > tree_.value[node][0][1]:
                result += f"{indent}y = 0.\n"
            else:
                result += f"{indent}y = 1.\n"
        return result

    return recurse(0, 0)


# ==================================================================================
# 5. PROMPT
# ==================================================================================
def build_prompt(rules, trees, scores, target_idx: int, use_tree: bool = True, top_k: int = 7) -> str:
    """[REUSED: gen_prompt()] Build the optimisation prompt, wording kept exactly as released.

    The top_k best rules so far are listed from worst to best with their score (validation
    accuracy x 10000) and, if use_tree is True, the decision tree fitted with that rule.
    [OURS] use_tree=False removes every mention of decision trees (score-only ablation).
    """
    order = np.argsort(np.array(scores))[-top_k:]
    idx = target_idx
    text = f"I have some rules to generate x{idx} from x1"
    for i in range(1, idx - 1):
        text += f", x{i + 1}"
    if use_tree:
        text += ". We also have corresponding decision tree (CART) to predict 'y' from x1"
        for i in range(1, idx):
            text += f", x{i + 1}"
    text += (". The rules are arragned in ascending order based on their scores evaluated with "
             "XGBoost classifier, where higher scores indicates better quality.")
    text += "\n\n"
    for i in order:
        text += f"Rule to generate x{idx}:\n{rules[i]}\n"
        if use_tree:
            text += f"Decision tree (CART):\n{trees[i]}"
        text += "Score evaluated with XGBoost classifier:\n{:.0f}".format(scores[i] * 10000)
        text += "\n\n"
    text += (f"Give me a new rule to generate x{idx} that is totally different from the old ones "
             f"and has a score as high as possible. ")
    if use_tree:
        text += (f"Decision trees (including both CART and XGBoost classifier) trained with newly "
                 f"generated x{idx} should be better than the old ones. ")
    else:
        text += f"XGBoost classifier trained with newly generated x{idx} should be better than the old ones. "
    text += f"Write the rule to generate x{idx} from x1"
    for i in range(1, idx - 1):
        text += f", x{i + 1}"
    text += (f" in square brackets. Variables x1 ~ x{idx} are in [0, 1]. You can use various numpy function. "
             f"Do not use np.log, np.sqrt, np.arcsin, np.arccos, np.arctan. Do not divide. When divide or "
             f"using log, use (x+1) term. Think creatively. The new rule must be written with Python grammar.")
    text += " Return the rule only with no explanation."
    return text


# ==================================================================================
# 6. PROPOSERS - anything that takes the prompt and returns text containing a rule
# ==================================================================================
class LLMUnavailable(RuntimeError):
    """The LLM could not be reached (rate limit, daily quota, network). Progress is saved."""


# OpenAI-compatible endpoints of providers that have offered free access. Base URLs and
# free limits change, so check the provider's own page before a long run.
PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": "OLLAMA_API_KEY"},
}


class ChatLLM:
    """[OURS] An LLM reached through an OpenAI-compatible chat API (Groq, Gemini, OpenRouter, Ollama ...).

    This replaces the authors' use_api(), which runs a local Llama 2 model. The sampling
    settings follow theirs: temperature 1.0 and top_p 0.95, with an empty system prompt.
    """

    def __init__(self, provider: str, model: str, api_key: str = None, base_url: str = None,
                 temperature: float = 1.0, top_p: float = 0.95, max_tokens: int = 4096,
                 min_seconds_between_calls: float = 0.0, max_retries: int = 6):
        from openai import OpenAI
        cfg = PROVIDERS.get(provider, {})
        model = model[len("models/"):] if model.startswith("models/") else model
        self.provider, self.model = provider, model
        base_url = base_url or cfg.get("base_url")
        api_key = api_key or os.environ.get(cfg.get("key_env", ""), "") or ("ollama" if provider == "ollama" else "")
        if not base_url:
            raise ValueError(f"Unknown provider '{provider}': pass base_url=...")
        if not api_key:
            raise ValueError(f"No API key: pass api_key=... or set {cfg.get('key_env', 'the key variable')}")
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens
        self.min_gap, self.max_retries = min_seconds_between_calls, max_retries
        self._last = 0.0
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self.seconds = 0.0
        self.name = f"{provider}:{model}"

    def list_models(self):
        """Model names your key can use (to pick a valid value for `model`)."""
        return sorted(m.id[len("models/"):] if m.id.startswith("models/") else m.id for m in self.client.models.list().data)

    def __call__(self, prompt: str, n_vars: int = None) -> str:
        import openai
        for attempt in range(self.max_retries + 1):
            wait = self.min_gap - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            t0 = time.time()
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens)
                self._last = time.time()
                self.seconds += self._last - t0
                self.calls += 1
                if getattr(r, "usage", None):
                    self.prompt_tokens += r.usage.prompt_tokens or 0
                    self.completion_tokens += r.usage.completion_tokens or 0
                return r.choices[0].message.content or ""
            except openai.RateLimitError as e:
                self._last = time.time()
                pause = 20.0 * (attempt + 1)
                try:                                          # use the server's own waiting time if given
                    pause = float(e.response.headers.get("retry-after", pause))
                except Exception:
                    pass
                if attempt == self.max_retries or pause > 900:
                    raise LLMUnavailable(f"Rate or daily limit reached on {self.name}: {str(e)[:200]}")
                print(f"    rate limit, waiting {pause:.0f}s ...")
                time.sleep(pause + 1)
            except (openai.APIConnectionError, openai.InternalServerError) as e:
                self._last = time.time()
                if attempt == self.max_retries:
                    raise LLMUnavailable(f"{self.name} unreachable: {str(e)[:200]}")
                time.sleep(5 * (attempt + 1))
        raise LLMUnavailable(f"{self.name} did not answer")


class HFLocalLLM:
    """[ADAPTED from load_model() and use_api()] The authors' way: a Llama model run locally on a GPU.

    Needs a GPU (a free Colab T4 is enough) and: pip install transformers accelerate bitsandbytes.
    Default model = the fine-tuned Llama 2 the authors used. Not tested by us on a GPU.
    """

    def __init__(self, model_path: str = "kykim0/Llama-2-7b-ultrachat200k-2e",
                 temperature: float = 1.0, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                   bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant,
                                                          torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.temperature, self.max_new_tokens = temperature, max_new_tokens
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self.seconds = 0.0
        self.name = f"hf_local:{model_path}"

    def __call__(self, prompt: str, n_vars: int = None) -> str:
        t0 = time.time()
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, do_sample=True, max_new_tokens=self.max_new_tokens,
                                      pad_token_id=self.tokenizer.eos_token_id, top_k=50, top_p=0.95,
                                      temperature=self.temperature, num_return_sequences=1)
        n_in = inputs.input_ids.shape[1]
        self.calls += 1
        self.prompt_tokens += int(n_in)
        self.completion_tokens += int(out.shape[1] - n_in)
        self.seconds += time.time() - t0
        return self.tokenizer.decode(out[0][n_in:], skip_special_tokens=True)


# ==================================================================================
# 7. THE OPTIMISATION LOOP
# ==================================================================================
def initial_rule(Xtr, ytr, Xva, yva) -> str:
    """[ADAPTED] Starting rule = product of the two columns XGBoost finds most important.

    As in the authors' script: default XGBoost with max_depth 1..11, keep the model with
    the best validation accuracy, and multiply its two most important columns.
    """
    best, best_val = None, -1.0
    for depth in range(1, 12):
        clf = XGBClassifier(max_depth=depth, tree_method="hist", device=xgb_device(),
                            random_state=0, seed=0, n_jobs=-1).fit(Xtr, ytr)
        val = accuracy_score(yva, clf.predict(Xva))
        if val > best_val:
            best, best_val = clf, val
    top = np.argsort(-best.feature_importances_)
    return "x{} = [x{} * x{}]".format(Xtr.shape[1] + 1, top[0] + 1, top[1] + 1)


def _best_index(vals, trains) -> int:
    """Best validation score; ties go to the lower training accuracy (the authors' tie-break)."""
    vals, trains = np.array(vals), np.array(trains)
    cand = np.where(vals == vals.max())[0]
    return int(cand[np.argmin(trains[cand])])


def _features(split, accepted):
    """Original columns plus every accepted generated column (each scaled on the training part)."""
    Xtr, Xva, Xte = split["xtrain"], split["xval"], split["xtest"]
    for feat in accepted:
        ctr, cva, cte = apply_rule(feat["rule"], Xtr), apply_rule(feat["rule"], Xva), apply_rule(feat["rule"], Xte)
        ctr, cva, cte, _, _ = scale_column(ctr, cva, cte)
        Xtr, Xva, Xte = np.c_[Xtr, ctr], np.c_[Xva, cva], np.c_[Xte, cte]
    return Xtr, Xva, Xte


def run_octree(seed: int, proposer, run_name: str, n_iters: int = 50, max_features: int = 2,
               use_tree: bool = True, cart_max_depth: int = 4, n_estimators=None,
               data_dir: str = "dataset", out_dir: str = "runs", max_invalid: int = 15,
               verbose: bool = True) -> dict:
    """Run OCTree on one split and return (and save) everything that happened.

    For each new feature (up to max_features):
      1. start from the initial rule (product of the two most important columns);
      2. repeat n_iters times: show the proposer the best rules so far with their scores
         (and decision trees if use_tree), get a new rule, add the column, train XGBoost,
         record the validation score and the decision tree;
      3. keep the rule with the best validation score - but only if it beats the model
         we had before this feature. Otherwise stop ('until the validation score no
         longer improves', Section 3.2 of the paper).
    Test accuracy is recorded for reporting only; it is never shown to the proposer and
    never used to choose a rule.

    The state is written to runs/<run_name>/seed<k>.json after every step, so if the LLM
    hits a limit you can call the function again with the same arguments and it resumes.
    """
    split = load_split(seed, data_dir)
    ytr, yva, yte = split["ytrain"], split["yval"], split["ytest"]
    path = os.path.join(out_dir, run_name, f"seed{seed}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    config = {"seed": seed, "n_iters": n_iters, "max_features": max_features, "use_tree": use_tree,
              "cart_max_depth": cart_max_depth, "n_estimators": n_estimators,
              "proposer": getattr(proposer, "name", type(proposer).__name__)}

    if os.path.exists(path):
        with open(path) as f:
            st = json.load(f)
        old = {k: st["config"].get(k) for k in config}
        same = lambda d: {k: v for k, v in d.items() if k != "n_iters"}
        if same(old) != same(config):
            raise ValueError(f"{path} was created with different settings:\n  saved {old}\n  now   {config}\n"
                             f"Use another run_name or delete the file.")
    else:
        st = {"config": config, "baseline": None, "rounds": [], "accepted": [], "finished": False,
              "timing": {"llm": 0.0, "xgboost": 0.0, "cart": 0.0, "rule": 0.0},
              "llm_calls": 0, "invalid_answers": 0, "prompt_tokens": 0, "completion_tokens": 0}
    # [OURS] n_iters may be changed between sessions (e.g. lowered if time runs short); a round
    # that already has n_iters proposals is then treated as complete.
    if st["config"].get("n_iters") != n_iters:
        if st["finished"]:
            n_iters = st["config"]["n_iters"]             # finished runs keep their own setting
        else:
            print(f"Note: proposals per feature changed from {st['config'].get('n_iters')} to {n_iters}.")
            st["config"]["n_iters"] = n_iters

    def save():
        with open(path, "w") as f:
            json.dump(st, f, indent=1)

    def log(msg):
        if verbose:
            print(msg, flush=True)

    def call_proposer(prompt, n_vars):
        before = (proposer.calls, proposer.prompt_tokens, proposer.completion_tokens, proposer.seconds)
        t0 = time.time()
        try:
            return proposer(prompt, n_vars)
        finally:
            st["timing"]["llm"] += time.time() - t0
            st["llm_calls"] += proposer.calls - before[0]
            st["prompt_tokens"] += proposer.prompt_tokens - before[1]
            st["completion_tokens"] += proposer.completion_tokens - before[2]

    def evaluate(rule, Xtr, Xva, Xte):
        """Add the rule's column, train XGBoost and CART; return one trajectory entry."""
        t0 = time.time()
        ctr, cva, cte = apply_rule(rule, Xtr), apply_rule(rule, Xva), apply_rule(rule, Xte)
        ctr, cva, cte, lo, hi = scale_column(ctr, cva, cte)
        Ntr, Nva, Nte = np.c_[Xtr, ctr], np.c_[Xva, cva], np.c_[Xte, cte]
        t1 = time.time()
        tr, va, te, _ = fit_and_score(Ntr, ytr, Nva, yva, Nte, yte, seed, n_estimators)
        t2 = time.time()
        cart = fit_cart(Ntr, ytr, Nva, yva, seed, cart_max_depth)
        text = tree_to_text(cart, [f"x{i}" for i in range(1, Ntr.shape[1] + 1)])
        t3 = time.time()
        st["timing"]["rule"] += t1 - t0
        st["timing"]["xgboost"] += t2 - t1
        st["timing"]["cart"] += t3 - t2
        return {"rule": rule, "train": tr, "val": va, "test": te, "tree": text, "col_min": lo, "col_max": hi}

    # ---- Step 0: baseline on the original columns -------------------------------
    if st["baseline"] is None:
        t0 = time.time()
        tr, va, te, _ = fit_and_score(split["xtrain"], ytr, split["xval"], yva, split["xtest"], yte, seed, n_estimators)
        st["timing"]["xgboost"] += time.time() - t0
        st["baseline"] = {"train": tr, "val": va, "test": te}
        save()
    log(f"[{run_name} | seed {seed}] Step 0 (original columns) | Val: {st['baseline']['val'] * 100:.2f} "
        f"| Test: {st['baseline']['test'] * 100:.2f}")

    # ---- generate features one after another -------------------------------------
    while not st["finished"]:
        Xtr, Xva, Xte = _features(split, st["accepted"])
        target = Xtr.shape[1] + 1
        incumbent = st["accepted"][-1] if st["accepted"] else st["baseline"]
        if not st["rounds"] or st["rounds"][-1]["done"]:
            if len(st["rounds"]) >= max_features:
                st["finished"] = True
                save()
                break
            st["rounds"].append({"target": target, "trajectory": [], "done": False})
        rnd = st["rounds"][-1]
        traj = rnd["trajectory"]

        if not traj:                                                 # iteration 0: the initial rule
            entry = evaluate(initial_rule(Xtr, ytr, Xva, yva), Xtr, Xva, Xte)
            entry["source"] = "initial"
            traj.append(entry)
            save()
            log(f"  x{target} | iter  0 | Val: {entry['val'] * 100:.2f} | Test: {entry['test'] * 100:.2f} | {entry['rule']}")

        while len(traj) - 1 < n_iters:
            prompt = build_prompt([e["rule"] for e in traj], [e["tree"] for e in traj],
                                  [e["val"] for e in traj], target, use_tree)
            entry, bad = None, 0
            while entry is None:
                if bad >= max_invalid:
                    save()
                    raise RuntimeError(f"{max_invalid} unusable answers in a row. Last answer:\n{answer[:500]}")
                try:
                    answer = call_proposer(prompt, target - 1)
                except LLMUnavailable:
                    save()
                    log("  LLM unavailable - progress saved. Run the same call again later to resume.")
                    raise
                rule = extract_rule(answer, target)
                try:
                    if rule is None:
                        raise InvalidRule("no rule found in the answer")
                    if rule in [e["rule"] for e in traj]:
                        raise InvalidRule("same rule as before")      # the paper asks for a rule not in the trajectory
                    entry = evaluate(rule, Xtr, Xva, Xte)
                except InvalidRule:
                    bad += 1
                    st["invalid_answers"] += 1
            entry["source"] = "proposed"
            traj.append(entry)
            save()
            best = traj[_best_index([e["val"] for e in traj], [e["train"] for e in traj])]
            log(f"  x{target} | iter {len(traj) - 1:2d} | Val: {entry['val'] * 100:.2f} | best Val: {best['val'] * 100:.2f} "
                f"(Test {best['test'] * 100:.2f}) | {entry['rule'][:90]}")

        best = traj[_best_index([e["val"] for e in traj], [e["train"] for e in traj])]
        rnd["done"] = True
        rnd["best_rule"] = best["rule"]
        if best["val"] > incumbent["val"]:
            st["accepted"].append({k: best[k] for k in ("rule", "train", "val", "test", "col_min", "col_max")})
            rnd["accepted"] = True
            log(f"  -> x{target} accepted: {best['rule']}  (Val {best['val'] * 100:.2f})")
        else:
            rnd["accepted"] = False
            st["finished"] = True
            log(f"  -> x{target} not accepted: validation did not improve. Stopping.")
        save()

    final = st["accepted"][-1] if st["accepted"] else st["baseline"]
    st["final"] = {"val": final["val"], "test": final["test"], "test_error": 100 * (1 - final["test"]),
                   "baseline_error": 100 * (1 - st["baseline"]["test"]),
                   "rules": [a["rule"] for a in st["accepted"]]}
    save()
    log(f"[{run_name} | seed {seed}] Final test error {st['final']['test_error']:.2f}% "
        f"(baseline {st['final']['baseline_error']:.2f}%) with {len(st['accepted'])} generated feature(s)")
    return st


def load_run(run_name: str, seed: int, out_dir: str = "runs"):
    """Read a saved run, or return None if it does not exist."""
    path = os.path.join(out_dir, run_name, f"seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ==================================================================================
# 8. FINAL MODEL AND INFERENCE  [OURS]
# ==================================================================================
def train_final_model(seed: int, rules, model_dir: str = "model", data_dir: str = "dataset",
                      n_estimators=None, extra: dict = None) -> dict:
    """Train XGBoost on the original + generated columns and save everything inference needs.

    Writes model/xgb_model.json (the trained model) and model/pipeline.json (how to turn
    eight raw inputs into the model's input: scaling values and the generated-feature rules).
    """
    split = load_split(seed, data_dir)
    Xtr, Xva, Xte = split["xtrain"], split["xval"], split["xtest"]
    feats = []
    for rule in rules:
        ctr, cva, cte = apply_rule(rule, Xtr), apply_rule(rule, Xva), apply_rule(rule, Xte)
        ctr, cva, cte, lo, hi = scale_column(ctr, cva, cte)
        feats.append({"rule": rule, "col_min": lo, "col_max": hi})
        Xtr, Xva, Xte = np.c_[Xtr, ctr], np.c_[Xva, cva], np.c_[Xte, cte]
    tr, va, te, clf = fit_and_score(Xtr, split["ytrain"], Xva, split["yval"], Xte, split["ytest"], seed, n_estimators)
    os.makedirs(model_dir, exist_ok=True)
    clf.save_model(os.path.join(model_dir, "xgb_model.json"))
    pipe = {"dataset": f"OpenML {OPENML_ID} (electricity)", "seed": seed,
            "raw_columns": RAW_COLUMNS, "raw_min": split["raw_min"].tolist(), "raw_max": split["raw_max"].tolist(),
            "generated_features": feats, "labels": {"0": "DOWN", "1": "UP"},
            "metrics": {"train_accuracy": tr, "val_accuracy": va, "test_accuracy": te,
                        "test_error_percent": 100 * (1 - te)}}
    # A few real test rows (raw values + true label) so the app can offer ready-made examples.
    rng = np.random.RandomState(0)
    pick = rng.choice(len(split["xtest"]), size=min(30, len(split["xtest"])), replace=False)
    raw = split["xtest"][pick] * (split["raw_max"] - split["raw_min"]) + split["raw_min"]
    pipe["examples"] = [{"values": [round(float(v), 6) for v in row], "true_label": pipe["labels"][str(int(lab))]}
                        for row, lab in zip(raw, split["ytest"][pick])]
    if extra:
        pipe.update(extra)
    with open(os.path.join(model_dir, "pipeline.json"), "w") as f:
        json.dump(pipe, f, indent=2)
    return pipe


def load_pipeline(model_dir: str = "model"):
    """Load the saved pipeline description and the trained XGBoost model."""
    with open(os.path.join(model_dir, "pipeline.json")) as f:
        pipe = json.load(f)
    clf = XGBClassifier()
    clf.load_model(os.path.join(model_dir, "xgb_model.json"))
    return pipe, clf


def preprocess_raw(pipe: dict, raw_rows) -> np.ndarray:
    """Input -> Preprocessing: scale raw values like the training data, then add generated columns."""
    X = np.atleast_2d(np.asarray(raw_rows, dtype=float))
    if X.shape[1] != len(pipe["raw_columns"]):
        raise ValueError(f"expected {len(pipe['raw_columns'])} values per row: {pipe['raw_columns']}")
    lo, hi = np.array(pipe["raw_min"]), np.array(pipe["raw_max"])
    X = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
    for feat in pipe["generated_features"]:
        col = apply_rule(feat["rule"], X)
        span = feat["col_max"] - feat["col_min"]
        X = np.c_[X, (col - feat["col_min"]) / (span if span > 0 else 1.0)]
    return X


def predict(pipe: dict, clf, raw_rows):
    """Preprocessing -> Trained model -> Prediction. Returns (labels, probability of UP)."""
    X = preprocess_raw(pipe, raw_rows)
    p_up = clf.predict_proba(X)[:, 1]
    labels = [pipe["labels"][str(int(p >= 0.5))] for p in p_up]
    return labels, p_up


# ==================================================================================
# 9. SUMMARIES  [OURS]
# ==================================================================================
def summarise(run_names, seeds=(0, 1, 2), out_dir: str = "runs"):
    """One row per run: mean and standard deviation of the test error over the seeds found."""
    import pandas as pd
    rows = []
    for name in run_names:
        runs = [r for r in (load_run(name, s, out_dir) for s in seeds) if isinstance(r, dict) and r.get("final")]
        if not runs:
            continue
        base = np.array([r["final"]["baseline_error"] for r in runs])
        err = np.array([r["final"]["test_error"] for r in runs])
        rows.append({"run": name, "seeds": len(runs),
                     "baseline error %": f"{base.mean():.2f} ± {base.std():.2f}",
                     "test error %": f"{err.mean():.2f} ± {err.std():.2f}",
                     "relative reduction %": round(100 * (base.mean() - err.mean()) / base.mean(), 1),
                     "features kept (per seed)": [len(r["final"]["rules"]) for r in runs],
                     "proposals": sum(r["llm_calls"] for r in runs),
                     "unusable answers": sum(r["invalid_answers"] for r in runs)})
    return pd.DataFrame(rows)


def best_so_far(run: dict, which: str = "val"):
    """For plotting: per feature round, the running best validation score and its test accuracy."""
    curves = []
    for rnd in run["rounds"]:
        vals, trains, tests = [], [], []
        best_val, best_test = [], []
        for e in rnd["trajectory"]:
            vals.append(e["val"]); trains.append(e["train"]); tests.append(e["test"])
            i = _best_index(vals, trains)
            best_val.append(vals[i]); best_test.append(tests[i])
        curves.append({"target": rnd["target"], "val": vals, "best_val": best_val, "best_test": best_test})
    return curves


# ==================================================================================
# 10. OUR EXPERIMENTS  [OURS]
#     C. Baseline comparison: hand-made features vs OCTree features
#     E. Data size: do generated features help more when training data is scarce?
# ==================================================================================
# Hand-made features, written from what the columns mean (the LLM never sees these names):
#   x1 date, x2 day, x3 period, x4 nswprice, x5 nswdemand, x6 vicprice, x7 vicdemand, x8 transfer
# Like OCTree's rules, each one uses only the values in the same row, so the comparison is fair.
MANUAL_FEATURES = [
    {"name": "price_gap", "expr": "x4 - x6",
     "why": "NSW price minus Victoria price: when NSW is dearer than its neighbour, prices tend to move"},
    {"name": "demand_gap", "expr": "x5 - x7",
     "why": "NSW demand minus Victoria demand: relative pressure on the two markets"},
    {"name": "price_x_demand", "expr": "x4 * x5",
     "why": "NSW price times NSW demand: high price under high demand signals a tight market"},
    {"name": "period_sin", "expr": "np.sin(2 * np.pi * x3)",
     "why": "time of day as a circle (23:30 sits next to 00:00), first coordinate"},
    {"name": "period_cos", "expr": "np.cos(2 * np.pi * x3)",
     "why": "time of day as a circle, second coordinate"},
    {"name": "peak_hours", "expr": "1.0 if 14 / 47 <= x3 < 44 / 47 else 0.0",
     "why": "1 during roughly 07:00-22:00 (half-hour slots 14-43), when demand and prices peak"},
    {"name": "transfer_x_price_gap", "expr": "x8 * (x4 - x6)",
     "why": "interstate transfer weighted by the price gap: flow in the direction of the dearer market"},
]


def add_rules(Xtr, Xva, Xte, rules):
    """Append generated columns one after another (each scaled with the training part's min/max).

    A later rule may use an earlier generated column (e.g. x10 may use x9), so the order matters.
    """
    for rule in rules:
        ctr, cva, cte = apply_rule(rule, Xtr), apply_rule(rule, Xva), apply_rule(rule, Xte)
        ctr, cva, cte, _, _ = scale_column(ctr, cva, cte)
        Xtr, Xva, Xte = np.c_[Xtr, ctr], np.c_[Xva, cva], np.c_[Xte, cte]
    return Xtr, Xva, Xte


def run_manual_features(seed: int, run_name: str, max_features: int = 2, n_estimators=None,
                        candidates=MANUAL_FEATURES, data_dir: str = "dataset", out_dir: str = "runs",
                        verbose: bool = True) -> dict:
    """Experiment C. Choose hand-made features with exactly OCTree's selection rule.

    Round 1 tries every hand-made feature as x9 and keeps the best one on validation data, but
    only if it beats the original columns. Round 2 tries the rest as x10 on top of it, and so on,
    up to max_features. The only difference from OCTree is where the candidate columns come from:
    a person who knows the column meanings, instead of an LLM that does not.
    A second result, 'all', adds every hand-made feature at once without any selection.
    Saved in the same format as OCTree runs, so summarise() can put them in one table.
    """
    path = os.path.join(out_dir, run_name, f"seed{seed}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    config = {"seed": seed, "max_features": max_features, "n_estimators": n_estimators,
              "candidates": [c["expr"] for c in candidates]}
    if os.path.exists(path):
        with open(path) as f:
            st = json.load(f)
        if st["config"] == config and st.get("final"):
            if verbose:
                print(f"[{run_name} | seed {seed}] already done: {st['final']['test_error']:.2f}% "
                      f"with {st['final']['rules'] or 'no features'}")
            return st

    s = load_split(seed, data_dir)
    ytr, yva, yte = s["ytrain"], s["yval"], s["ytest"]
    t_start = time.time()

    def score(rules):
        Xtr, Xva, Xte = add_rules(s["xtrain"], s["xval"], s["xtest"], rules)
        tr, va, te, _ = fit_and_score(Xtr, ytr, Xva, yva, Xte, yte, seed, n_estimators)
        return {"train": tr, "val": va, "test": te}

    base = score([])
    st = {"config": config, "baseline": base, "rounds": [], "accepted": [],
          "llm_calls": 0, "invalid_answers": 0, "prompt_tokens": 0, "completion_tokens": 0}
    if verbose:
        print(f"[{run_name} | seed {seed}] original columns | Val: {base['val'] * 100:.2f} | Test: {base['test'] * 100:.2f}")
    chosen, incumbent, remaining = [], base, list(candidates)
    while remaining and len(chosen) < max_features:
        target = 9 + len(chosen)
        tried = []
        for c in remaining:
            rule = f"x{target} = [{c['expr']}]"
            r = score([a["rule"] for a in chosen] + [rule])
            r.update(rule=rule, name=c["name"])
            tried.append(r)
            if verbose:
                print(f"  x{target} | {c['name']:22s} | Val: {r['val'] * 100:.2f} | Test: {r['test'] * 100:.2f}")
        best = tried[_best_index([t["val"] for t in tried], [t["train"] for t in tried])]
        accepted = best["val"] > incumbent["val"]
        st["rounds"].append({"target": target, "tried": tried, "best": best["name"], "accepted": accepted})
        if not accepted:
            if verbose:
                print(f"  -> no hand-made feature improves validation accuracy. Stopping.")
            break
        chosen.append(best)
        incumbent = best
        remaining = [c for c in remaining if c["name"] != best["name"]]
        if verbose:
            print(f"  -> x{target} accepted: {best['name']}  (Val {best['val'] * 100:.2f})")
    st["accepted"] = chosen
    st["final"] = {"val": incumbent["val"], "test": incumbent["test"], "test_error": 100 * (1 - incumbent["test"]),
                   "baseline_error": 100 * (1 - base["test"]), "rules": [c["rule"] for c in chosen],
                   "names": [c["name"] for c in chosen]}
    all_rules = [f"x{9 + k} = [{c['expr']}]" for k, c in enumerate(candidates)]
    r_all = score(all_rules)
    st["all"] = {"rules": all_rules, "val": r_all["val"], "test": r_all["test"], "test_error": 100 * (1 - r_all["test"])}
    st["seconds"] = time.time() - t_start
    with open(path, "w") as f:
        json.dump(st, f, indent=1)
    if verbose:
        print(f"[{run_name} | seed {seed}] selected: {st['final']['test_error']:.2f}% | "
              f"all {len(candidates)} at once: {st['all']['test_error']:.2f}% | baseline {st['final']['baseline_error']:.2f}%")
    return st


def run_data_size(seed: int, feature_sets: dict, run_name: str, fractions=(0.1, 0.25, 0.5, 1.0),
                  n_estimators=None, data_dir: str = "dataset", out_dir: str = "runs", verbose: bool = True):
    """Experiment E. Train on a growing share of the training rows, with and without generated features.

    feature_sets maps a name to a list of rules, e.g. {"original": [], "OCTree": [...], "hand-made": [...]}.
    The smaller training sets are nested (the 10% rows are inside the 25% rows, and so on) and drawn
    at random with a fixed seed. Validation and test sets are always the full ones, so every result
    is measured on the same rows. Each finished (fraction, feature set) is saved, so the cell resumes.
    """
    path = os.path.join(out_dir, run_name, f"seed{seed}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            done = {(r["fraction"], r["features"], tuple(r["rules"])): r for r in json.load(f)}
    s = load_split(seed, data_dir)
    order = np.random.RandomState(100 + seed).permutation(len(s["ytrain"]))
    rows = []
    for frac in fractions:
        idx = np.sort(order[: max(50, int(round(frac * len(order))))])     # keep the original row order
        for name, rules in feature_sets.items():
            key = (float(frac), name, tuple(rules))
            if key in done:
                rows.append(done[key])
                continue
            t0 = time.time()
            Xtr, Xva, Xte = add_rules(s["xtrain"][idx], s["xval"], s["xtest"], rules)
            tr, va, te, _ = fit_and_score(Xtr, s["ytrain"][idx], Xva, s["yval"], Xte, s["ytest"], seed, n_estimators)
            r = {"seed": seed, "fraction": float(frac), "train_rows": int(len(idx)), "features": name,
                 "rules": list(rules), "val": va, "test": te, "test_error": 100 * (1 - te),
                 "seconds": time.time() - t0}
            rows.append(r)
            done[key] = r
            with open(path, "w") as f:
                json.dump(list(done.values()), f, indent=1)
            if verbose:
                print(f"[seed {seed}] {frac:>5.0%} of training rows ({len(idx):5d}) | {name:10s} | "
                      f"test error {r['test_error']:.2f}%")
    return rows
