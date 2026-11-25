"""
app.py - Streamlit web app for the trained OCTree + XGBoost electricity model.  [OURS]

Pipeline shown to the user:
    Input -> Preprocessing -> Trained model -> Prediction -> Output

Start it with:   streamlit run app.py      (opens at http://localhost:8501)
It needs the folder  model/  (xgb_model.json and pipeline.json), which the notebook
(Section 8) or train.py --save_model creates. No LLM or internet is needed to predict.
"""
import io
import os

import numpy as np
import pandas as pd
import streamlit as st

import octree_core as oc

MODEL_DIR = os.environ.get("OCTREE_MODEL_DIR", "model")

# Friendly names and help texts for the eight inputs (the model itself only sees x1 ... x8).
NICE = {"date": "Date", "day": "Day of week", "period": "Time of day",
        "nswprice": "NSW price", "nswdemand": "NSW demand",
        "vicprice": "Victoria price", "vicdemand": "Victoria demand", "transfer": "Interstate transfer"}
HELP = {
    "date": "Date on the dataset's 0-1 scale (0 = start of the recorded period, 1 = end).",
    "day": "Day-of-week code used in the dataset (0-6).",
    "period": "Half-hour of the day: slot 0 starts at 00:00, slot 47 at 23:30.",
    "nswprice": "New South Wales electricity price, on the dataset's normalised scale.",
    "nswdemand": "New South Wales electricity demand, normalised.",
    "vicprice": "Victoria electricity price, normalised.",
    "vicdemand": "Victoria electricity demand, normalised.",
    "transfer": "Scheduled electricity transfer between the two states, normalised.",
}
TIME_COLS = ["date", "day", "period"]
MARKET_COLS = ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]

st.set_page_config(page_title="Electricity Price Direction", page_icon="⚡", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero h1 {font-size: 2.1rem; margin-bottom: .2rem;}
.hero p {color: #4B5563; font-size: 1.02rem; margin-top: 0;}
.pipeline {display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; margin: .6rem 0 1.2rem 0;}
.step {background: #EEF2FF; color: #1E3A8A; border: 1px solid #C7D2FE; border-radius: 999px;
       padding: .28rem .8rem; font-size: .86rem; font-weight: 600;}
.arrow {color: #9CA3AF; font-weight: 700;}
.card {border: 1px solid #E5E7EB; border-radius: 14px; padding: 1.1rem 1.3rem; background: #FFFFFF;}
.result {border-radius: 14px; padding: 1.2rem 1.4rem; color: #FFFFFF; margin-bottom: .8rem;}
.result.up {background: linear-gradient(135deg, #15803D, #22A05A);}
.result.down {background: linear-gradient(135deg, #1D4ED8, #3B6FE6);}
.result .label {font-size: 2.3rem; font-weight: 800; letter-spacing: .04em; line-height: 1.1;}
.result .sub {opacity: .92; font-size: .98rem; margin-top: .2rem;}
.bar {height: 12px; border-radius: 999px; background: #DBEAFE; overflow: hidden; margin: .35rem 0 .2rem 0;}
.bar > div {height: 100%; background: #16A34A;}
.barlabels {display: flex; justify-content: space-between; font-size: .8rem; color: #6B7280;}
.truth-ok {color: #15803D; font-weight: 600;}
.truth-bad {color: #B91C1C; font-weight: 600;}
.small {color: #6B7280; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ model
@st.cache_resource
def load():
    return oc.load_pipeline(MODEL_DIR)


if not os.path.exists(os.path.join(MODEL_DIR, "pipeline.json")):
    st.error(f"No trained model found in '{MODEL_DIR}/'. Run the notebook (Section 8) or train.py --save_model first, "
             f"then start the app from the project folder.")
    st.stop()

pipe, clf = load()
cols = pipe["raw_columns"]
examples = pipe.get("examples", [])
metrics = pipe["metrics"]
features = pipe["generated_features"]


def readable(rule: str) -> str:
    """'x9 = [x4 * x1]' -> 'NSW price × Date' (real column names, for people only)."""
    expr = oc.rule_expression(rule)
    names = {f"x{i + 1}": NICE[c] for i, c in enumerate(cols)}
    for k in range(len(cols) + 1, len(cols) + 1 + len(features)):
        names[f"x{k}"] = f"generated x{k}"
    for k in sorted(names, key=len, reverse=True):
        expr = expr.replace(k, names[k])
    return expr.replace(" * ", " × ")


def slot_time(slot: int) -> str:
    return f"{slot // 2:02d}:{30 * (slot % 2):02d}"


# ------------------------------------------------------------------ sidebar: model card
with st.sidebar:
    st.header("Model")
    st.metric("Test error", f"{metrics['test_error_percent']:.2f}%")
    st.metric("Test accuracy", f"{metrics['test_accuracy'] * 100:.2f}%")
    st.caption(f"{pipe['dataset']} · split seed {pipe['seed']} · XGBoost")
    st.subheader("Generated features")
    if features:
        for feat in features:
            st.code(feat["rule"], language="python")
            st.caption(f"= {readable(feat['rule'])}")
    else:
        st.write("None (baseline model).")
    st.caption("x1…x8 are the scaled inputs: " + ", ".join(cols))

# ------------------------------------------------------------------ header
st.markdown("""
<div class="hero">
  <h1>⚡ Electricity Price Direction</h1>
  <p>Predicts whether the New South Wales electricity price is <b>UP</b> or <b>DOWN</b> compared with its
  average over the last 24 hours. The model is XGBoost trained on the eight original columns plus features
  generated with <b>OCTree</b> (an LLM proposes feature rules, guided by decision-tree feedback).</p>
</div>
<div class="pipeline">
  <span class="step">1 · Input</span><span class="arrow">→</span>
  <span class="step">2 · Preprocessing</span><span class="arrow">→</span>
  <span class="step">3 · Trained model</span><span class="arrow">→</span>
  <span class="step">4 · Prediction</span><span class="arrow">→</span>
  <span class="step">5 · Output</span>
</div>
""", unsafe_allow_html=True)

tab_one, tab_many, tab_about = st.tabs(["Single prediction", "CSV file", "How it works"])

# ------------------------------------------------------------------ tab 1: single prediction
with tab_one:
    # Starting values for the input widgets: the first saved test example (or mid-range values).
    defaults = dict(zip(cols, examples[0]["values"])) if examples else {c: 0.5 for c in cols}

    def to_widget(c, v):
        if c == "period":
            return int(round(v * 47))
        if c == "day":
            return int(round(v))
        return float(v)

    for c in cols:
        st.session_state.setdefault("in_" + c, to_widget(c, defaults[c]))

    def use_example():
        ex = examples[np.random.randint(len(examples))]
        for c, v in zip(cols, ex["values"]):
            st.session_state["in_" + c] = to_widget(c, v)
        st.session_state["example"] = ex
        st.session_state.pop("result", None)

    top_left, top_right = st.columns([3, 2])
    with top_left:
        st.markdown("#### 1 · Enter the half-hour you want to predict")
    with top_right:
        if examples:
            st.button("Load a real example from the test set", on_click=use_example, width="stretch")

    with st.form("inputs"):
        left, right = st.columns(2)
        values = {}
        with left:
            st.markdown("**When**")
            values["date"] = st.number_input(NICE["date"], min_value=0.0, max_value=1.0, step=0.01,
                                             format="%.4f", key="in_date", help=HELP["date"])
            values["day"] = st.selectbox(NICE["day"] + " (code 0-6)", list(range(7)), key="in_day", help=HELP["day"])
            slot = st.slider(NICE["period"] + " (half-hour slot 0-47)", 0, 47, key="in_period", help=HELP["period"])
            st.caption(f"Slot {slot} starts at {slot_time(slot)}")
            values["period"] = slot / 47
        with right:
            st.markdown("**Market (normalised values)**")
            for c in MARKET_COLS:
                values[c] = st.number_input(NICE[c], min_value=0.0, max_value=1.5, step=0.01, format="%.4f",
                                            key="in_" + c, help=HELP[c])
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if submitted:
        row = [float(values[c]) for c in cols]
        try:
            labels, p_up = oc.predict(pipe, clf, [row])
            X = oc.preprocess_raw(pipe, [row])[0]
        except Exception as e:                       # e.g. a generated feature cannot be computed for these inputs
            st.session_state.pop("result", None)
            st.error(f"Could not compute a prediction for these inputs: {e}")
        else:
            ex = st.session_state.get("example")
            same = ex is not None and np.allclose(row, ex["values"], atol=1e-6)
            st.session_state["result"] = {"row": row, "label": labels[0], "p": float(p_up[0]), "X": X.tolist(),
                                          "truth": ex["true_label"] if same else None}

    res = st.session_state.get("result")
    if res:
        st.markdown("#### 2 · Result")
        out_left, out_right = st.columns([3, 2])
        with out_left:
            up = res["label"] == "UP"
            sub = ("The price is expected to be above its 24-hour average." if up
                   else "The price is expected to be below its 24-hour average.")
            st.markdown(f'<div class="result {"up" if up else "down"}"><div class="label">{res["label"]}</div>'
                        f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bar"><div style="width:{res["p"] * 100:.1f}%"></div></div>'
                        f'<div class="barlabels"><span>Probability of UP: <b>{res["p"]:.1%}</b></span>'
                        f'<span>Probability of DOWN: <b>{1 - res["p"]:.1%}</b></span></div>', unsafe_allow_html=True)
        with out_right:
            if res["truth"]:
                ok = res["truth"] == res["label"]
                st.markdown(f'<div class="card">True label of this test example: <b>{res["truth"]}</b><br>'
                            f'<span class="{"truth-ok" if ok else "truth-bad"}">'
                            f'{"✓ The model is correct" if ok else "✗ The model is wrong on this one"}</span></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="card">These are your own input values, so there is no true label to '
                            'compare with.<br><span class="small">Load a test example to compare with the real '
                            'answer.</span></div>', unsafe_allow_html=True)

        with st.expander("What happened to the input (preprocessing steps)"):
            X = res["X"]
            names = [f"x{i + 1}" for i in range(len(X))]
            meaning = [NICE[c] for c in cols] + [f"generated: {readable(f['rule'])}" for f in features]
            raw = [round(v, 6) for v in res["row"]] + [None] * len(features)
            st.dataframe(pd.DataFrame({"model input": names, "meaning": meaning, "raw value": raw,
                                       "value given to the model": np.round(X, 4)}),
                         hide_index=True, width="stretch")
            st.caption("Raw values are min-max scaled with the training data's minimum and maximum; the generated "
                       "features are computed from the scaled values with the saved rules and scaled the same way.")

# ------------------------------------------------------------------ tab 2: batch prediction
with tab_many:
    st.markdown("#### Predict many rows at once")
    st.write("Upload a CSV file with these eight columns (same scale as the dataset): `" + "`, `".join(cols) + "`.")
    if examples:
        template = pd.DataFrame([ex["values"] for ex in examples[:5]], columns=cols)
        st.download_button("Download a template CSV (5 real test rows)", template.to_csv(index=False),
                           "template.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read this file as CSV: {e}")
            df = None
        if df is not None:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            elif df.empty:
                st.warning("The file has no rows.")
            else:
                data = df[cols].apply(pd.to_numeric, errors="coerce")
                bad = data.isna().any(axis=1)
                if bad.any():
                    rows_bad = ", ".join(str(i + 2) for i in np.where(bad)[0][:10])
                    st.error(f"Some rows contain empty or non-numeric values (file lines {rows_bad}"
                             f"{' …' if bad.sum() > 10 else ''}). Fix them and upload again.")
                else:
                    try:
                        labels, p_up = oc.predict(pipe, clf, data.to_numpy(dtype=float))
                    except Exception as e:
                        st.error(f"Could not compute predictions for this file: {e}")
                    else:
                        out = df.copy()
                        out["prediction"], out["probability_UP"] = labels, np.round(p_up, 4)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Rows", f"{len(out):,}")
                        c2.metric("Predicted UP", f"{int((out['prediction'] == 'UP').sum()):,}")
                        c3.metric("Predicted DOWN", f"{int((out['prediction'] == 'DOWN').sum()):,}")
                        st.dataframe(out.head(200), hide_index=True, width="stretch")
                        if len(out) > 200:
                            st.caption(f"Showing the first 200 of {len(out):,} rows; the download has all of them.")
                        buf = io.StringIO()
                        out.to_csv(buf, index=False)
                        st.download_button("Download predictions", buf.getvalue(), "predictions.csv", "text/csv",
                                           type="primary")

# ------------------------------------------------------------------ tab 3: how it works
with tab_about:
    st.markdown("#### How this prediction is made")
    st.markdown(
        "1. **Input** – eight values describing one half-hour of the NSW electricity market.\n"
        "2. **Preprocessing** – each value is min-max scaled with the minimum and maximum of the *training* data, "
        "and the OCTree feature rules add new columns.\n"
        "3. **Trained model** – an XGBoost classifier trained on the training rows only, with the original "
        "authors' tuned settings.\n"
        "4. **Prediction** – the model gives the probability that the price is UP.\n"
        "5. **Output** – UP if that probability is at least 50 %, otherwise DOWN.")
    st.markdown("#### Generated features (found by OCTree)")
    if features:
        st.dataframe(pd.DataFrame({"rule": [f["rule"] for f in features],
                                   "meaning (real column names)": [readable(f["rule"]) for f in features]}),
                     hide_index=True, width="stretch")
    else:
        st.write("None: this model uses only the original eight columns.")
    st.markdown("#### Model facts")
    facts = [("Dataset", pipe["dataset"]),
             ("Split", f"random 60/20/20, seed {pipe['seed']}"),
             ("Test accuracy", f"{metrics['test_accuracy'] * 100:.2f}%"),
             ("Test error", f"{metrics['test_error_percent']:.2f}%"),
             ("Validation accuracy", f"{metrics['val_accuracy'] * 100:.2f}%"),
             ("Run", str(pipe.get("run", "-"))),
             ("LLM used to find the features", str(pipe.get("proposer", "-")))]
    st.markdown("| Item | Value |\n|---|---|\n" + "\n".join(f"| **{k}** | {v} |" for k, v in facts))
    st.caption("No LLM or internet connection is needed to predict: the feature rules are stored in "
               "model/pipeline.json. The LLM was only used during training to find them.")

st.markdown('<p class="small" style="margin-top:2rem">Machine Learning (101700) · Reproduction of OCTree '
            '(Nam et al., NeurIPS 2024) on the electricity dataset</p>', unsafe_allow_html=True)
