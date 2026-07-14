import streamlit as st
import pandas as pd
import json
import re
import spacy

from spacy.matcher import PhraseMatcher
from collections import Counter
from wordcloud import WordCloud

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Product Review Tagging",
    page_icon="📱",
    layout="wide"
)

# =========================
# LOAD SPACY MODEL
# =========================

nlp = spacy.load("en_core_web_sm")

# =========================
# TITLE
# =========================

st.title("📱 Product Review Tagging Dashboard")
st.write("NewtonAI Internship Project")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("📱 Product Review Tagging")

st.sidebar.info("""
Developer: Manan Dureja

Technology Stack:
• Python
• Streamlit
• spaCy
• Pandas

Project:
Rule-Based NLP Review Tagging
""")

# =========================
# LOAD TAG DICTIONARY
# =========================

with open("tag_dictionary.json", "r") as f:
    tag_dictionary = json.load(f)

# =========================
# CREATE MATCHER
# =========================

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

for tag, phrases in tag_dictionary.items():
    patterns = [nlp.make_doc(p) for p in phrases]
    matcher.add(tag, patterns)

# =========================
# FUNCTIONS
# =========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

def tag_review(text):
    doc = nlp(text)

    matches = matcher(doc)

    tags = set()

    for match_id, start, end in matches:
        tags.add(nlp.vocab.strings[match_id])

    return list(tags)

def sentiment(rating):

    if rating >= 4:
        return "Positive"

    elif rating == 3:
        return "Neutral"

    else:
        return "Negative"

# =========================
# LIVE REVIEW ANALYZER
# =========================

st.subheader("🤖 Live Review Analyzer")

user_review = st.text_area("Enter a review")

if st.button("Analyze Review"):

    detected_tags = tag_review(
        clean_text(user_review)
    )

    st.success(
        f"Detected Tags: {detected_tags}"
    )

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload Reviews CSV",
    type=["csv"]
)

# =========================
# MAIN DASHBOARD
# =========================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    required_columns = ["Review", "Rating"]

    if not all(col in df.columns for col in required_columns):

        st.error(
            "CSV must contain Review and Rating columns."
        )

    else:

        # ---------------------
        # PROCESS DATA
        # ---------------------

        df["Clean_Review"] = df["Review"].apply(clean_text)

        df["Tags"] = df["Clean_Review"].apply(tag_review)

        df["Sentiment"] = df["Rating"].apply(sentiment)

        all_tags = []

        for tag_list in df["Tags"]:
            all_tags.extend(tag_list)

        tag_counts = Counter(all_tags)

        tag_df = pd.DataFrame(
            tag_counts.items(),
            columns=["Tag", "Count"]
        )

        # ---------------------
        # METRICS
        # ---------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Reviews",
            len(df)
        )

        col2.metric(
            "Average Rating",
            round(df["Rating"].mean(), 2)
        )

        col3.metric(
            "Unique Tags",
            len(tag_df)
        )

        # ---------------------
        # DATASET PREVIEW
        # ---------------------

        st.subheader("📄 Dataset Preview")

        st.dataframe(df.head())

        # ---------------------
        # TAGGED REVIEWS
        # ---------------------

        st.subheader("🏷️ Tagged Reviews")

        st.dataframe(
            df[
                [
                    "Review",
                    "Rating",
                    "Tags",
                    "Sentiment"
                ]
            ]
        )

        # ---------------------
        # TAG DISTRIBUTION
        # ---------------------

        st.subheader("📊 Tag Distribution")

        if not tag_df.empty:

            st.dataframe(tag_df)

            st.bar_chart(
                tag_df.set_index("Tag")
            )

        # ---------------------
        # SENTIMENT ANALYSIS
        # ---------------------

        st.subheader("😊 Sentiment Analysis")

        sentiment_counts = (
            df["Sentiment"]
            .value_counts()
        )

        st.bar_chart(sentiment_counts)

        # ---------------------
        # FILTER
        # ---------------------

        st.subheader("🔍 Filter Reviews")

        selected = st.selectbox(
            "Filter by Sentiment",
            [
                "All",
                "Positive",
                "Neutral",
                "Negative"
            ]
        )

        if selected == "All":
            st.dataframe(df)
        else:
            st.dataframe(
                df[df["Sentiment"] == selected]
            )

        # ---------------------
        # TOP ISSUES
        # ---------------------

        st.subheader("🚨 Top Customer Issues")

        if not tag_df.empty:

            top5 = (
                tag_df
                .sort_values(
                    by="Count",
                    ascending=False
                )
                .head(5)
            )

            st.dataframe(top5)

        # ---------------------
        # WORD CLOUD
        # ---------------------

        st.subheader("☁️ Word Cloud")

        text = " ".join(
            df["Clean_Review"]
            .astype(str)
        )

        if text.strip():

            wordcloud = WordCloud(
                width=800,
                height=400
            ).generate(text)

            st.image(
                wordcloud.to_array()
            )

        # ---------------------
        # BUSINESS INSIGHTS
        # ---------------------

        st.subheader("💡 Business Insights")

        if not tag_df.empty:

            top_issue = (
                tag_df
                .sort_values(
                    by="Count",
                    ascending=False
                )
                .iloc[0]
            )

            st.write(
                f"Most Common Issue: {top_issue['Tag']}"
            )

            st.write(
                f"Mentions: {top_issue['Count']}"
            )

            st.write(
                "Recommendation: Improve this area to increase customer satisfaction."
            )

        # ---------------------
        # DOWNLOAD
        # ---------------------

        csv = df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download Tagged Reviews",
            csv,
            "tagged_reviews.csv",
            "text/csv"
        )

        # ---------------------
        # FUTURE SCOPE
        # ---------------------

        st.subheader("🚀 Future Scope")

        st.markdown("""
- LLM-Based Review Classification
- BERT Sentiment Analysis
- Multilingual Support
- Real-Time Review Monitoring
- AI Recommendation Engine
""")

        # ---------------------
        # ARCHITECTURE
        # ---------------------

        st.subheader("📋 Project Architecture")

        st.code("""
Reviews
   ↓
Preprocessing
   ↓
Phrase Matching
   ↓
Tag Generation
   ↓
Sentiment Analysis
   ↓
Business Insights
""")