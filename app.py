import streamlit as st
import pickle
import re
import string
import pandas as pd

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 35px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

.true-news {
    background-color: #e8f7ee;
    border: 2px solid #2e8b57;
    color: #176b3a;
}

.fake-news {
    background-color: #fdeaea;
    border: 2px solid #d9534f;
    color: #a52824;
}

.model-card {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    border: 1px solid #ddd;
    text-align: center;
}

textarea {
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    with open("vectorizer.pkl", "rb") as f:
        vectorization = pickle.load(f)

    with open("lr_model.pkl", "rb") as f:
        LR = pickle.load(f)

    with open("dt_model.pkl", "rb") as f:
        DT = pickle.load(f)

    with open("gbc_model.pkl", "rb") as f:
        GBC = pickle.load(f)

    with open("rfc_model.pkl", "rb") as f:
        RFC = pickle.load(f)

    return vectorization, LR, DT, GBC, RFC


vectorization, LR, DT, GBC, RFC = load_models()


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def wordopt(text):

    text = text.lower()

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(
        r'https?://\S+|www\.\S+',
        '',
        text
    )

    text = re.sub(
        r'<.*?>',
        '',
        text
    )

    text = re.sub(
        r'[%s]' % re.escape(string.punctuation),
        '',
        text
    )

    text = re.sub(
        r'\n',
        ' ',
        text
    )

    text = re.sub(
        r'\w*\d\w*',
        '',
        text
    )

    return text


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">📰 Fake News Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered news classification using Machine Learning & NLP'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ About the Project")

    st.write(
        """
        This application uses Machine Learning
        to classify news articles as:

        **🟢 True News**

        or

        **🔴 Fake News**
        """
    )

    st.divider()

    st.subheader("Models Used")

    st.write("• Logistic Regression")
    st.write("• Decision Tree")
    st.write("• Gradient Boosting")
    st.write("• Random Forest")

    st.divider()

    st.info(
        "The model predicts based on patterns learned "
        "from the training dataset. It is not an independent "
        "fact-checking system."
    )


# =========================================================
# INPUT AREA
# =========================================================

st.subheader("🔍 Enter News Article")

news = st.text_area(
    "Paste the complete news article below:",
    height=250,
    placeholder="Paste your news article here..."
)


# =========================================================
# CHECK BUTTON
# =========================================================

if st.button(
    "🚀 Check News",
    use_container_width=True
):

    if news.strip() == "":

        st.warning("⚠️ Please enter a news article first.")

    else:

        # Clean text
        cleaned_news = wordopt(news)

        # Convert to dataframe
        test_df = pd.DataFrame({
            "text": [cleaned_news]
        })

        # TF-IDF transformation
        news_vector = vectorization.transform(
            test_df["text"]
        )

        # Predictions
        pred_lr = LR.predict(news_vector)[0]
        pred_dt = DT.predict(news_vector)[0]
        pred_gbc = GBC.predict(news_vector)[0]
        pred_rfc = RFC.predict(news_vector)[0]

        # Convert labels
        def label(pred):
            if pred == 0:
                return "Fake News"
            else:
                return "True News"

        lr_result = label(pred_lr)
        dt_result = label(pred_dt)
        gbc_result = label(pred_gbc)
        rfc_result = label(pred_rfc)

        # =================================================
        # FINAL RESULT
        # =================================================

        predictions = [
            pred_lr,
            pred_dt,
            pred_gbc,
            pred_rfc
        ]

        true_count = sum(
            1 for p in predictions if p == 1
        )

        fake_count = sum(
            1 for p in predictions if p == 0
        )

        if true_count >= fake_count:

            st.markdown(
                """
                <div class="result-box true-news">
                    <h1>✅ TRUE NEWS</h1>
                    <p>Most models classified this article as True News.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-box fake-news">
                    <h1>❌ FAKE NEWS</h1>
                    <p>Most models classified this article as Fake News.</p>
                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # MODEL PREDICTIONS
        # =================================================

        st.subheader("🤖 Model Predictions")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Logistic Regression",
                lr_result
            )

        with col2:
            st.metric(
                "Decision Tree",
                dt_result
            )

        with col3:
            st.metric(
                "Gradient Boosting",
                gbc_result
            )

        with col4:
            st.metric(
                "Random Forest",
                rfc_result
            )


        # =================================================
        # SUMMARY
        # =================================================

        st.subheader("📊 Prediction Summary")

        result_data = pd.DataFrame({
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "Gradient Boosting",
                "Random Forest"
            ],

            "Prediction": [
                lr_result,
                dt_result,
                gbc_result,
                rfc_result
            ]
        })

        st.dataframe(
            result_data,
            use_container_width=True,
            hide_index=True
        )