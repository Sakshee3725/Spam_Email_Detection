
import streamlit as st
import pandas as pd
from simple_spam_detector import model, vectorizer


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="SpamShield | Email Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0b1120 0%, #111827 55%, #172554 100%);
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #263449;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(120deg, #172554, #312e81, #1e3a8a);
    border: 1px solid #3b4b91;
    margin-bottom: 24px;
}

.hero h1 {
    color: white;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #dbeafe;
    font-size: 15px;
    margin-bottom: 0;
}

.section-title {
    color: #f8fafc;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 8px;
}

.metric-card {
    background: #172033;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 20px;
    min-height: 120px;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.metric-value {
    color: #f8fafc;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

.result-card {
    padding: 24px;
    border-radius: 18px;
    background: #172033;
    border: 1px solid #334155;
    min-height: 190px;
}

.result-spam {
    border: 1px solid #ef4444;
    background: linear-gradient(135deg, #3b111c, #241725);
}

.result-safe {
    border: 1px solid #22c55e;
    background: linear-gradient(135deg, #102d26, #142b2d);
}

.result-title {
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 10px;
}

.result-description {
    color: #cbd5e1;
    font-size: 14px;
    line-height: 1.7;
}

div.stButton > button {
    border-radius: 10px;
    min-height: 43px;
    font-weight: 600;
    border: 1px solid #475569;
    transition: 0.2s;
}

div.stButton > button:hover {
    border-color: #818cf8;
    color: #c7d2fe;
}

.stTextArea textarea {
    background: #111827;
    color: #f8fafc;
    border: 1px solid #475569;
    border-radius: 12px;
}

div[data-testid="stExpander"] {
    border: 1px solid #334155;
    border-radius: 12px;
}

.footer {
    color: #94a3b8;
    text-align: center;
    padding: 22px 0 8px 0;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE ----------------
if "message" not in st.session_state:
    st.session_state.message = ""

if "result" not in st.session_state:
    st.session_state.result = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "scans" not in st.session_state:
    st.session_state.scans = 0

if "spam_count" not in st.session_state:
    st.session_state.spam_count = 0

if "scan_error" not in st.session_state:
    st.session_state.scan_error = None


# ---------------- SAMPLE EMAILS ----------------
SPAM_EMAIL = (
    "Congratulations! You have won a free $500 gift card. "
    "Claim your prize now by clicking this link immediately!"
)

SAFE_EMAIL = (
    "Dear Team,\n\n"
    "Please find attached the project report for your review. "
    "Kindly share your feedback before Friday.\n\n"
    "Best regards,\n"
    "Project Team"
)


# ---------------- CALLBACK FUNCTIONS ----------------
def load_spam_sample():
    st.session_state.message = SPAM_EMAIL
    st.session_state.result = None
    st.session_state.probabilities = None
    st.session_state.scan_error = None


def load_safe_sample():
    st.session_state.message = SAFE_EMAIL
    st.session_state.result = None
    st.session_state.probabilities = None
    st.session_state.scan_error = None


def clear_message():
    st.session_state.message = ""
    st.session_state.result = None
    st.session_state.probabilities = None
    st.session_state.scan_error = None


def scan_email():
    email_text = st.session_state.message.strip()

    if not email_text:
        st.session_state.scan_error = "Please enter an email message before scanning."
        st.session_state.result = None
        st.session_state.probabilities = None
        return

    try:
        email_vector = vectorizer.transform([email_text])
        prediction = int(model.predict(email_vector)[0])
        probabilities = model.predict_proba(email_vector)[0]

        # Classes are 0 = Safe and 1 = Spam
        class_probabilities = {
            int(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        }

        safe_probability = class_probabilities.get(0, 0.0) * 100
        spam_probability = class_probabilities.get(1, 0.0) * 100

        st.session_state.result = {
            "prediction": prediction,
            "safe_probability": safe_probability,
            "spam_probability": spam_probability,
        }

        st.session_state.probabilities = {
            "Safe Email": safe_probability,
            "Spam Email": spam_probability,
        }

        st.session_state.scans += 1

        if prediction == 1:
            st.session_state.spam_count += 1

        st.session_state.scan_error = None

    except Exception as error:
        st.session_state.scan_error = (
            f"Unable to scan this email. Please check the model files. "
            f"Details: {error}"
        )
        st.session_state.result = None
        st.session_state.probabilities = None


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 🛡️ SpamShield")
    st.caption("Email Security Dashboard")
    st.divider()

    st.markdown("### Navigation")
    st.markdown("🏠 **Email Scanner**")
    st.markdown("📊 **Detection Analytics**")
    st.markdown("🔍 **Model Information**")

    st.divider()

    st.markdown("### How it works")
    st.write(
        "1. Enter an email message.\n\n"
        "2. Click **Scan Email**.\n\n"
        "3. Review the predicted category and probabilities."
    )

    st.divider()
    st.caption("Built with Python, Streamlit and scikit-learn")


# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🛡️ Intelligent Email Spam Detector</h1>
    <p>
        Analyze email messages, identify potential spam, and review
        model confidence through a simple security dashboard.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- METRICS ----------------
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Emails Scanned</div>
        <div class="metric-value">{st.session_state.scans}</div>
    </div>
    """, unsafe_allow_html=True)

with metric2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Spam Detected</div>
        <div class="metric-value">{st.session_state.spam_count}</div>
    </div>
    """, unsafe_allow_html=True)

with metric3:
    safe_count = st.session_state.scans - st.session_state.spam_count

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Safe</div>
        <div class="metric-value">{safe_count}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ---------------- MAIN SCANNER LAYOUT ----------------
left_column, right_column = st.columns([1.15, 0.85], gap="large")

with left_column:
    st.markdown(
        '<div class="section-title">✉️ Email Scanner</div>',
        unsafe_allow_html=True
    )

    st.write("Paste an email message below to check its predicted category.")

    st.text_area(
        "Email message",
        key="message",
        height=220,
        placeholder=(
            "Example: Dear Team, please find the project report "
            "attached for your review..."
        ),
        label_visibility="collapsed"
    )

    button1, button2, button3 = st.columns(3)

    with button1:
        st.button(
            "🚨 Load Spam",
            use_container_width=True,
            on_click=load_spam_sample
        )

    with button2:
        st.button(
            "✉️ Load Safe",
            use_container_width=True,
            on_click=load_safe_sample
        )

    with button3:
        st.button(
            "🗑️ Clear",
            use_container_width=True,
            on_click=clear_message
        )

    st.write("")

    st.button(
        "🔍  SCAN EMAIL",
        type="primary",
        use_container_width=True,
        on_click=scan_email
    )

    if st.session_state.scan_error:
        st.error(st.session_state.scan_error)


# ---------------- RESULTS ----------------
with right_column:
    st.markdown(
        '<div class="section-title">📋 Scan Results</div>',
        unsafe_allow_html=True
    )

    result = st.session_state.result

    if result is None:
        st.markdown("""
        <div class="result-card">
            <div class="result-title">🔎 Ready to Analyze</div>
            <div class="result-description">
                Your scan results will appear here.<br><br>
                Enter an email message and click <b>SCAN EMAIL</b>
                to view the prediction and probability scores.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif result["prediction"] == 1:
        st.markdown(f"""
        <div class="result-card result-spam">
            <div class="result-title">🚨 SPAM DETECTED</div>
            <div class="result-description">
                This email has been classified as potential spam.<br><br>
                <b>Spam Probability:</b>
                {result["spam_probability"]:.1f}%<br>
                <b>Safe Probability:</b>
                {result["safe_probability"]:.1f}%<br><br>
                Be cautious with suspicious links, unexpected prizes,
                and requests for personal information.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-card result-safe">
            <div class="result-title">✅ SAFE EMAIL</div>
            <div class="result-description">
                This email has been classified as likely safe by the model.<br><br>
                <b>Safe Probability:</b>
                {result["safe_probability"]:.1f}%<br>
                <b>Spam Probability:</b>
                {result["spam_probability"]:.1f}%<br><br>
                Always check the sender and links before trusting an email.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------- PROBABILITY CHART ----------------
st.write("")
st.divider()

st.markdown(
    '<div class="section-title">📊 Detection Analytics</div>',
    unsafe_allow_html=True
)

if st.session_state.probabilities is not None:
    chart_data = pd.DataFrame({
        "Category": ["Safe Email", "Spam Email"],
        "Probability (%)": [
            st.session_state.probabilities["Safe Email"],
            st.session_state.probabilities["Spam Email"]
        ]
    }).set_index("Category")

    st.bar_chart(
        chart_data,
        y="Probability (%)",
        horizontal=True
    )

    st.caption(
        "Probabilities are estimates from the trained model, "
        "not guarantees that an email is safe or spam."
    )
else:
    st.info("Scan an email to display its probability chart.")


# ---------------- ABOUT SECTION ----------------
with st.expander("ℹ️ About this project"):
    st.markdown("""
    **SpamShield — Email Spam Detection System**

    This project uses a machine-learning text classification pipeline
    built with Python and scikit-learn.

    **Components**
    - CountVectorizer: converts email text into numerical features.
    - Multinomial Naive Bayes: predicts whether a message is spam or safe.
    - Streamlit: provides the interactive web dashboard.

    **Important limitation:** Predictions depend on the quality and size
    of the training dataset. A small sample dataset is suitable for a
    demonstration, but it cannot establish real-world accuracy or
    guarantee reliable detection for every email.
    """)


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    SpamShield | Python • Streamlit • scikit-learn
    <br>
    Educational project — predictions may not always be correct.
</div>
""", unsafe_allow_html=True)

#cmd = python -m streamlit run app.py