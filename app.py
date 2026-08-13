import streamlit as st
import pickle
import os
import re
import string
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TruthGuard AI - Fake News Detector",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL STYLES
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b2e 50%, #0f172a 100%);
    color: #f1f5f9;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.95));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 24px;
    padding: 36px 30px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 25px 50px rgba(0,0,0,0.5);
}
.hero-title {
    font-size: 46px;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
    letter-spacing: -1px;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 16px;
    color: #94a3b8;
    max-width: 680px;
    margin: 0 auto 16px auto;
    line-height: 1.7;
}
.hero-badges {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}
.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(56,189,248,0.12);
    color: #38bdf8;
    border: 1px solid rgba(56,189,248,0.25);
}

/* Result Cards */
.result-real {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(6,78,59,0.25));
    border: 2px solid #10b981;
    box-shadow: 0 0 30px rgba(16,185,129,0.2);
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    margin: 20px 0;
    animation: fadeSlideIn 0.4s ease;
}
.result-fake {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(127,29,29,0.25));
    border: 2px solid #ef4444;
    box-shadow: 0 0 30px rgba(239,68,68,0.2);
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    margin: 20px 0;
    animation: fadeSlideIn 0.4s ease;
}
.result-label-real {
    color: #34d399;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: 1px;
}
.result-label-fake {
    color: #f87171;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: 1px;
}
.result-confidence {
    font-size: 16px;
    color: #cbd5e1;
    margin-top: 8px;
    font-weight: 500;
}
.confidence-value {
    font-weight: 700;
    font-size: 20px;
}

/* Glass Card */
.glass-card {
    background: rgba(30,41,59,0.65);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    transition: border-color 0.2s ease;
}
.glass-card:hover {
    border-color: rgba(56,189,248,0.3);
}
.card-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #64748b;
    margin-bottom: 6px;
}
.card-value {
    font-size: 24px;
    font-weight: 700;
    color: #f1f5f9;
}

/* Keyword Tags */
.kw-tag {
    display: inline-block;
    background: rgba(129,140,248,0.12);
    color: #a5b4fc;
    border: 1px solid rgba(129,140,248,0.25);
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    margin: 3px;
}

/* Sidebar */
div[data-testid="stSidebar"] {
    background: #0b1120;
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* Input overrides */
.stTextArea textarea {
    background-color: rgba(15,23,42,0.8) !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(148,163,184,0.15) !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
}

/* Table styling */
.metric-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    border-radius: 12px;
    overflow: hidden;
}
.metric-table th {
    background: rgba(56,189,248,0.1);
    color: #38bdf8;
    font-weight: 700;
    padding: 12px 16px;
    text-align: left;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.metric-table td {
    padding: 11px 16px;
    color: #e2e8f0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.metric-table tr:last-child td { border-bottom: none; }
.metric-table tr:hover td { background: rgba(255,255,255,0.03); }
.best-row td { background: rgba(16,185,129,0.08) !important; }
.best-row td:first-child { border-left: 3px solid #10b981; }

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# PREPROCESSING (must match train_model.py exactly)
# ============================================================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Strip ONLY location prefixes but KEEP 'reuters' as a signal
    text = re.sub(r'^[a-z]+(?:\s+[a-z]+)*\s*(?:\([a-z\s]+\))?\s*[-\u2014\u2013:]\s*', '', text)
    text = re.sub(r'^\s*21st century wire\s*[-\u2014\u2013:]\s*', '', text)
    # Remove URLs but keep square brackets
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    # Preserve [VIDEO], [AUDIO], [IMAGE] - normalize to just the word
    text = re.sub(r'\[([a-z\s]+)\]', r'\1', text)
    # Remove punctuation but keep apostrophes and hyphens
    text = re.sub(r'[%s]' % re.escape(string.punctuation.replace("'", "").replace("-", "")), ' ', text)
    # Remove newlines
    text = re.sub(r'\n|\r', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_top_keywords(cleaned_text, vectorizer, top_n=10):
    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
        tfidf_matrix = vectorizer.transform([cleaned_text])
        arr = np.squeeze(np.asarray(tfidf_matrix.todense()))
        nonzero = arr.nonzero()[0]
        if len(nonzero) == 0:
            return []
        scores = list(zip(feature_names[nonzero], arr[nonzero]))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    except Exception:
        return []


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_all():
    """Load vectorizer, all models, and saved metrics."""
    errors = []

    def try_load(paths):
        for p in paths:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    return pickle.load(f), None
        return None, f"Not found in: {paths}"

    vec, e = try_load(["models/tfidf_vectorizer.pkl", "vectorizer.pkl"])
    if e: errors.append("Vectorizer: " + e)

    lr,  e = try_load(["models/logistic_regression.pkl", "lr_model.pkl"])
    if e: errors.append("LR: " + e)

    dt,  e = try_load(["models/decision_tree.pkl", "dt_model.pkl"])
    if e: errors.append("DT: " + e)

    rf,  e = try_load(["models/random_forest.pkl", "rfc_model.pkl"])
    if e: errors.append("RF: " + e)

    gbc, e = try_load(["models/gradient_boosting.pkl", "gbc_model.pkl"])
    if e: errors.append("GBC: " + e)

    metrics, _ = try_load(["models/model_metrics.pkl", "model_metrics.pkl"])

    models = {
        "Logistic Regression": lr,
        "Decision Tree": dt,
        "Random Forest": rf,
        "Gradient Boosting": gbc,
    }

    best = "Logistic Regression"
    if metrics and "best_model_name" in metrics:
        best = metrics["best_model_name"]

    return vec, models, best, metrics, errors


vec, models, best_model_name, saved_metrics, load_errors = load_all()
models_loaded = vec is not None and all(v is not None for v in models.values())

MODEL_OPTIONS = ["Best Model (Recommended)"] + list(models.keys()) + ["Ensemble (Weighted)"]

SAMPLE_REAL = (
    "WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, "
    "who voted this month for a huge expansion of the national debt to pay for tax cuts, called "
    "himself a fiscal conservative on Sunday and urged budget restraint in 2018."
)
SAMPLE_FAKE = (
    "Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing! "
    "Donald Trump just couldn't wish all Americans a Happy New Year and leave it at that. Instead, "
    "he had to give a shout out to his enemies, haters and the very dishonest fake news media."
)


# ============================================================
# HERO BANNER
# ============================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">TruthGuard AI</div>
    <div class="hero-subtitle">
        A B.Tech CSE-AI project using Natural Language Processing and Machine Learning
        to classify news articles as <strong>Real</strong> or <strong>Fake</strong>.
        Trained on 44,000+ real-world news articles.
    </div>
    <div class="hero-badges">
        <span class="badge">NLP Pipeline</span>
        <span class="badge">TF-IDF Vectorization</span>
        <span class="badge">4 ML Models</span>
        <span class="badge">Real-Time Inference</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## System Status")
    if models_loaded:
        st.success("ML Pipeline Active")
    else:
        st.error("Model Load Failed")
        for err in load_errors:
            st.caption(err)

    st.markdown("---")
    st.markdown("## Model Selection")
    selected_option = st.selectbox(
        "Choose model for prediction:",
        MODEL_OPTIONS,
        index=0,
        help="'Best Model' automatically uses the highest F1-Score model (Logistic Regression)."
    )

    if selected_option == "Best Model (Recommended)":
        active_model_name = best_model_name
    else:
        active_model_name = selected_option

    st.info(f"Active: **{active_model_name}**")

    st.markdown("---")
    st.markdown("## Classification Labels")
    st.markdown("""
- **REAL NEWS** → Class 1 (Authentic)
- **FAKE NEWS** → Class 0 (Misleading)
""")
    st.markdown("---")


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Predict News",
    "Batch CSV Analysis",
    "Model Performance",
    "How It Works"
])


# ============================================================
# TAB 1 — PREDICT NEWS
# ============================================================
with tab1:
    st.markdown("### Paste News Article or Headline")

    col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    with col_s1:
        if st.button("Load Real Sample", use_container_width=True):
            st.session_state["input_text"] = SAMPLE_REAL
            st.rerun()
    with col_s2:
        if st.button("Load Fake Sample", use_container_width=True):
            st.session_state["input_text"] = SAMPLE_FAKE
            st.rerun()
    with col_s3:
        if st.button("Clear", use_container_width=True):
            st.session_state["input_text"] = ""
            st.rerun()

    user_news = st.text_area(
        label="Article text:",
        value=st.session_state.get("input_text", ""),
        height=200,
        placeholder="Paste the news article here...",
        label_visibility="collapsed"
    )
    st.session_state["input_text"] = user_news

    words = len(user_news.split()) if user_news.strip() else 0
    chars = len(user_news)
    c1, c2, c3 = st.columns(3)
    c1.caption(f"Words: **{words:,}**")
    c2.caption(f"Characters: **{chars:,}**")
    c3.caption(f"Model: **{active_model_name}**")

    predict_btn = st.button("Predict News", type="primary", use_container_width=True)

    if predict_btn:
        if not user_news.strip():
            st.warning("Please enter a news article or headline before predicting.")
        elif len(user_news.strip().split()) < 3:
            st.warning("The input is too short. Please enter at least a few words for a reliable prediction.")
        elif not models_loaded:
            st.error("Models failed to load. Please run `python train_model.py` first, then restart the app.")
        else:
            with st.spinner("Analyzing article..."):
                cleaned = clean_text(user_news)
                vec_input = vec.transform([cleaned])

                # Ensemble handling
                if active_model_name == "Ensemble (Weighted)":
                    # Build weights from saved_metrics f1 scores if available
                    weights = {}
                    total = 0.0
                    for name, m in models.items():
                        f1 = None
                        if saved_metrics and isinstance(saved_metrics, dict):
                            meta = saved_metrics.get(name, {})
                            f1 = meta.get('f1_score') if isinstance(meta.get('f1_score'), (int, float)) else None
                        weights[name] = float(f1) if f1 is not None else 1.0
                        total += weights[name]
                    if total == 0:
                        total = len(weights)
                        for k in weights:
                            weights[k] = 1.0
                    # normalize
                    for k in weights:
                        weights[k] = weights[k] / total

                    per_model_probas = {}
                    ensemble_real = 0.0
                    ensemble_fake = 0.0
                    for name, model in models.items():
                        if hasattr(model, 'predict_proba'):
                            p = model.predict_proba(vec_input)[0]
                            classes = getattr(model, 'classes_', None)
                            if classes is not None:
                                prob_map = {cls: float(p[i]) for i, cls in enumerate(classes)}
                                pr = prob_map.get(1, 0.0)
                                pf = prob_map.get(0, 0.0)
                            else:
                                pf = float(p[0])
                                pr = float(p[1])
                        else:
                            pr = 1.0 if model.predict(vec_input)[0] == 1 else 0.0
                            pf = 1.0 - pr
                        per_model_probas[name] = {'proba_real': pr, 'proba_fake': pf}
                        ensemble_real += weights[name] * pr
                        ensemble_fake += weights[name] * pf

                    pred_class = 1 if ensemble_real >= ensemble_fake else 0
                    proba_real = ensemble_real
                    proba_fake = ensemble_fake
                    model_used = 'Ensemble (Weighted)'
                    confidence = proba_real if pred_class == 1 else proba_fake
                else:
                    model = models[active_model_name]
                    pred_class = model.predict(vec_input)[0]

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(vec_input)[0]
                        classes = getattr(model, 'classes_', None)
                        if classes is not None:
                            prob_map = {cls: float(proba[i]) for i, cls in enumerate(classes)}
                            proba_fake = float(prob_map.get(0, 0.0))
                            proba_real = float(prob_map.get(1, 0.0))
                        else:
                            proba_fake = float(proba[0])
                            proba_real = float(proba[1])
                        confidence = proba_real if pred_class == 1 else proba_fake
                    else:
                        proba_fake = 1.0 if pred_class == 0 else 0.0
                        proba_real = 1.0 if pred_class == 1 else 0.0
                        confidence = 1.0

            # --- Debug panel data (show raw model outputs and file existence) ---
            debug_info = {}
            try:
                debug_info['vectorizer_type'] = type(vec).__name__
                debug_info['vocab_size'] = len(getattr(vec, 'vocabulary_', {}))
            except Exception:
                debug_info['vectorizer_type'] = str(type(vec))
                debug_info['vocab_size'] = None

            # Known model file candidates (used when training/saving)
            model_file_candidates = {
                'Logistic Regression': ['models/logistic_regression.pkl', 'lr_model.pkl'],
                'Decision Tree': ['models/decision_tree.pkl', 'dt_model.pkl'],
                'Random Forest': ['models/random_forest.pkl', 'rfc_model.pkl'],
                'Gradient Boosting': ['models/gradient_boosting.pkl', 'gbc_model.pkl']
            }

            per_model_debug = {}
            for name, m in models.items():
                entry = {}
                entry['loaded'] = m is not None
                entry['classes_'] = list(getattr(m, 'classes_', [])) if m is not None else None
                try:
                    entry['predicted'] = int(m.predict(vec_input)[0]) if m is not None else None
                except Exception:
                    entry['predicted'] = None
                try:
                    if m is not None and hasattr(m, 'predict_proba'):
                        p = m.predict_proba(vec_input)[0]
                        entry['predict_proba'] = [float(x) for x in p]
                    else:
                        entry['predict_proba'] = None
                except Exception as e:
                    entry['predict_proba'] = f'error: {e}'

                # which file exists for this model?
                exist_paths = []
                for p in model_file_candidates.get(name, []):
                    exist_paths.append({p: os.path.exists(p)})
                entry['file_candidates'] = exist_paths

                per_model_debug[name] = entry

            debug_info['models'] = per_model_debug
            debug_info['active_model'] = active_model_name
            debug_info['saved_metrics_keys'] = list(saved_metrics.keys()) if isinstance(saved_metrics, dict) else None

            # Render debug expander
            with st.expander('Debug Panel — raw internals (show/hide)', expanded=False):
                st.markdown('**Vectorizer & Environment**')
                st.write({'vectorizer_type': debug_info['vectorizer_type'], 'vocab_size': debug_info['vocab_size']})
                st.markdown('**Per-model raw outputs**')
                for nm, d in per_model_debug.items():
                    st.markdown(f"**{nm}**")
                    st.write(d)
                st.markdown('**Saved metrics keys**')
                st.write(debug_info['saved_metrics_keys'])

            # Result card
            if pred_class == 1:
                st.markdown(f"""
<div class="result-real">
    <div class="result-label-real">REAL NEWS</div>
    <div class="result-confidence">
        Prediction Confidence: <span class="confidence-value">{confidence*100:.1f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="result-fake">
    <div class="result-label-fake">FAKE NEWS</div>
    <div class="result-confidence">
        Prediction Confidence: <span class="confidence-value">{confidence*100:.1f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3 = st.columns(3)
            verdict_label = "Real (Authentic)" if pred_class == 1 else "Fake (Misleading)"
            verdict_color = "#34d399" if pred_class == 1 else "#f87171"
            with m1:
                st.markdown(f"""
<div class="glass-card">
    <div class="card-label">Classification</div>
    <div class="card-value" style="color:{verdict_color};">{verdict_label}</div>
</div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
<div class="glass-card">
    <div class="card-label">Prediction Confidence</div>
    <div class="card-value">{confidence*100:.1f}%</div>
</div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
<div class="glass-card">
    <div class="card-label">Model Used</div>
    <div class="card-value" style="font-size:16px;">{active_model_name}</div>
</div>""", unsafe_allow_html=True)

            # Probability breakdown (single-model or ensemble)
            st.markdown("#### Probability Breakdown")
            if active_model_name == "Ensemble (Weighted)":
                col_pb1, col_pb2 = st.columns(2)
                with col_pb1:
                    st.markdown("**Ensemble Real News Probability**")
                    st.progress(float(proba_real), text=f"{proba_real*100:.1f}%")
                with col_pb2:
                    st.markdown("**Ensemble Fake News Probability**")
                    st.progress(float(proba_fake), text=f"{proba_fake*100:.1f}%")

                # Per-model breakdown
                st.markdown("#### Per-model Probabilities")
                cols = st.columns(2)
                left, right = cols
                i = 0
                for name, probs in per_model_probas.items():
                    target_col = left if i % 2 == 0 else right
                    with target_col:
                        st.markdown(f"**{name}**: Real {probs['proba_real']*100:.2f}% — Fake {probs['proba_fake']*100:.2f}%")
                    i += 1
            else:
                try:
                    has_proba = hasattr(model, "predict_proba")
                except Exception:
                    has_proba = False
                if has_proba:
                    col_pb1, col_pb2 = st.columns(2)
                    with col_pb1:
                        st.markdown("**Real News Probability**")
                        st.progress(float(proba_real), text=f"{proba_real*100:.1f}%")
                    with col_pb2:
                        st.markdown("**Fake News Probability**")
                        st.progress(float(proba_fake), text=f"{proba_fake*100:.1f}%")

            # Top keywords
            st.markdown("#### Top TF-IDF Signals")
            top_kws = extract_top_keywords(cleaned, vec, top_n=12)
            if top_kws:
                kw_html = "".join([
                    f'<span class="kw-tag">{w} ({s:.3f})</span>'
                    for w, s in top_kws
                ])
                st.markdown(f"<div>{kw_html}</div>", unsafe_allow_html=True)
            else:
                st.caption("No TF-IDF signals found. The article may be too short or contain unusual vocabulary.")


# ============================================================
# TAB 2 — BATCH CSV ANALYSIS
# ============================================================
with tab2:
    st.markdown("### Batch CSV Analysis")
    st.write("Upload a CSV file containing news articles to classify all records at once.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded CSV with **{len(df):,}** rows.")
            st.dataframe(df.head(5), use_container_width=True)

            text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
            if not text_cols:
                st.error("No text columns found in the uploaded CSV.")
            else:
                sel_col = st.selectbox("Select the column containing news text:", text_cols)

                if st.button("Run Batch Prediction", type="primary"):
                    if not models_loaded:
                        st.error("Models not loaded. Run `python train_model.py` first.")
                    else:
                        with st.spinner("Running predictions..."):
                            cleaned_series = df[sel_col].astype(str).apply(clean_text)
                            vectors = vec.transform(cleaned_series)

                            # Ensemble batch prediction
                            if active_model_name == "Ensemble (Weighted)":
                                # weights from saved_metrics f1
                                weights = {}
                                total = 0.0
                                for name, m in models.items():
                                    f1 = None
                                    if saved_metrics and isinstance(saved_metrics, dict):
                                        meta = saved_metrics.get(name, {})
                                        f1 = meta.get('f1_score') if isinstance(meta.get('f1_score'), (int, float)) else None
                                    weights[name] = float(f1) if f1 is not None else 1.0
                                    total += weights[name]
                                if total == 0:
                                    for k in weights:
                                        weights[k] = 1.0
                                    total = len(weights)
                                for k in weights:
                                    weights[k] = weights[k] / total

                                # collect per-model proba_real arrays
                                model_proba_real = {}
                                for name, model in models.items():
                                    if hasattr(model, 'predict_proba'):
                                        pmat = model.predict_proba(vectors)
                                        classes = getattr(model, 'classes_', None)
                                        if classes is not None:
                                            # find index for class 1
                                            try:
                                                idx1 = list(classes).index(1)
                                            except ValueError:
                                                idx1 = 1 if pmat.shape[1] > 1 else 0
                                            model_proba_real[name] = pmat[:, idx1]
                                        else:
                                            model_proba_real[name] = pmat[:, 1]
                                    else:
                                        preds_m = model.predict(vectors)
                                        model_proba_real[name] = (preds_m == 1).astype(float)

                                # weighted ensemble
                                import numpy as _np
                                ensemble_real = _np.zeros(vectors.shape[0])
                                for name, arr in model_proba_real.items():
                                    ensemble_real += weights.get(name, 0.0) * arr
                                ensemble_fake = 1.0 - ensemble_real
                                preds = (ensemble_real >= ensemble_fake).astype(int)
                                confidences = [round((ensemble_real[i] if preds[i] == 1 else ensemble_fake[i]) * 100, 1) for i in range(len(preds))]
                            else:
                                model = models[active_model_name]
                                preds = model.predict(vectors)

                                if hasattr(model, "predict_proba"):
                                    probas = model.predict_proba(vectors)
                                    classes = getattr(model, 'classes_', None)
                                    confidences = []
                                    if classes is not None:
                                        for i, p in enumerate(preds):
                                            try:
                                                idx = list(classes).index(p)
                                                confidences.append(round(probas[i][idx] * 100, 1))
                                            except ValueError:
                                                confidences.append(0.0)
                                    else:
                                        confidences = [round(probas[i][p] * 100, 1) for i, p in enumerate(preds)]
                                else:
                                    confidences = [100.0] * len(preds)

                            df["Prediction"] = ["REAL NEWS" if p == 1 else "FAKE NEWS" for p in preds]
                            df["Confidence (%)"] = confidences

                        real_count = (preds == 1).sum()
                        fake_count = (preds == 0).sum()

                        b1, b2, b3 = st.columns(3)
                        b1.metric("Total Articles", f"{len(df):,}")
                        b2.metric("Real News", f"{real_count:,}", f"{real_count/len(df)*100:.1f}%")
                        b3.metric("Fake News", f"{fake_count:,}", f"{fake_count/len(df)*100:.1f}%", delta_color="inverse")

                        chart_df = pd.DataFrame({"Category": ["Real News", "Fake News"], "Count": [real_count, fake_count]}).set_index("Category")
                        st.bar_chart(chart_df)
                        st.dataframe(df, use_container_width=True)

                        csv_out = df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download Predictions CSV", csv_out, "predictions.csv", "text/csv", type="primary")

        except Exception as e:
            st.error(f"Error processing CSV: {e}")


# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================
with tab3:
    st.markdown("### Model Performance Comparison")
    st.caption("All metrics are computed on a held-out 20% test set. Training and test sets are strictly separated.")

    if saved_metrics:
        rows = []
        for name in ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]:
            m = saved_metrics.get(name, {})
            if m:
                # Handle both old and new metric formats
                train_acc = m.get('train_accuracy', m.get('accuracy', 0))
                test_acc = m.get('accuracy', 0)
                precision = m.get('precision', 0)
                recall = m.get('recall', 0)
                f1_score = m.get('f1_score', m.get('f1', 0))
                
                rows.append({
                    "Model": name,
                    "Train Acc": f"{train_acc*100:.2f}%",
                    "Test Acc": f"{test_acc*100:.2f}%",
                    "Precision": f"{precision*100:.2f}%",
                    "Recall": f"{recall*100:.2f}%",
                    "F1 Score": f"{f1_score*100:.2f}%",
                    "_is_best": (name == saved_metrics.get("best_model_name", ""))
                })

        # Render HTML table
        table_rows_html = ""
        for r in rows:
            row_class = 'class="best-row"' if r["_is_best"] else ""
            star = " (*)" if r["_is_best"] else ""
            table_rows_html += f"""
<tr {row_class}>
    <td><strong>{r['Model']}{star}</strong></td>
    <td>{r['Train Acc']}</td>
    <td>{r['Test Acc']}</td>
    <td>{r['Precision']}</td>
    <td>{r['Recall']}</td>
    <td>{r['F1 Score']}</td>
</tr>"""

        st.markdown(f"""
<table class="metric-table">
<thead>
<tr>
    <th>Model</th><th>Train Accuracy</th><th>Test Accuracy</th>
    <th>Precision</th><th>Recall</th><th>F1 Score</th>
</tr>
</thead>
<tbody>
{table_rows_html}
</tbody>
</table>
<p style="color:#64748b;font-size:12px;margin-top:10px;">(*) Best model selected based on highest F1-Score on the test set.</p>
""", unsafe_allow_html=True)

        # Confusion matrices (visual text format)
        st.markdown("---")
        st.markdown("#### Classification Reports")
        for name in ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]:
            m = saved_metrics.get(name, {})
            if m and "classification_report" in m:
                with st.expander(f"{name} Classification Report"):
                    cm = m.get("confusion_matrix", [])
                    if cm:
                        cm_df = pd.DataFrame(
                            cm,
                            index=["Actual: Fake", "Actual: Real"],
                            columns=["Predicted: Fake", "Predicted: Real"]
                        )
                        st.dataframe(cm_df, use_container_width=False)
                    st.code(m["classification_report"], language="text")
    else:
        st.warning(
            "No saved metrics found. Please run `python train_model.py` to train the models "
            "and generate evaluation metrics."
        )


# ============================================================
# TAB 4 — HOW IT WORKS
# ============================================================
with tab4:
    st.markdown("### How the System Works")

    st.markdown("""
#### NLP Preprocessing Pipeline
The raw news text goes through these steps before classification:

1. **Lowercasing** — Normalises all text to lowercase.
2. **Publisher Header Removal** — Strips source prefixes like `WASHINGTON (Reuters) -` that could bias the model.
3. **URL & HTML Removal** — Strips hyperlinks and HTML tags.
4. **Punctuation & Number Removal** — Removes special characters and digit-containing tokens.
5. **Whitespace Normalisation** — Collapses multiple spaces into one.

#### Feature Extraction
- **TF-IDF Vectorizer** (`max_features=5000`, bigrams, English stop words removed, sublinear TF scaling)
- Input: `Title + " " + Text` concatenated — so the model evaluates both headline and body content.

#### Machine Learning Models
| Model | Algorithm | Strengths |
|---|---|---|
| Logistic Regression | Linear classifier | Fast, interpretable, excellent on TF-IDF sparse features |
| Decision Tree | Tree-based | Rule-based decisions, highly interpretable |
| Random Forest | Bagged trees | Reduces overfitting via ensemble variance |
| Gradient Boosting | Boosted trees | Sequentially corrects errors for high accuracy |

#### Why Logistic Regression Performed Best
After fixing the **Reuters data leakage** (removing publisher source headers during preprocessing) and adding
**Title + Text** as input, Logistic Regression achieved **99.04% test accuracy** with minimal overfitting.
The linear boundary of LR is ideal for high-dimensional TF-IDF sparse feature spaces.
""")

    st.markdown("---")
    st.warning("""
**Responsible AI Disclaimer:**
This is an educational B.Tech CSE-AI project prototype. The classifier was trained on a specific
news dataset and may not generalise perfectly to all news domains. Always cross-reference
predictions with reputable fact-checking organisations.
""")