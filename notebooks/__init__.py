import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import re
from urllib.parse import urlparse
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ---------------------- NLTK Setup ---------------------- #
custom_nltk_data_path = r"C:/10 Kifia Tasks/Week-1/financial-news-stock-analysis/notebooks/nltk_data"
nltk.data.path.append(custom_nltk_data_path)

nltk.download("punkt", download_dir=custom_nltk_data_path)
nltk.download("stopwords", download_dir=custom_nltk_data_path)

stop_words = set(stopwords.words("english"))
financial_terms = ["FDA approval", "price target", "merger", "downgrade", "earnings", "guidance", "valuation", "buyout"]
punctuation_regex = re.compile(r"[{}]".format(re.escape(string.punctuation)))  # Precompiled regex for punctuation removal

# ---------------------- Load Data ---------------------- #
df = pd.read_csv(r'C:/10 Kifia Tasks/data/raw_analyst_ratings.csv')

### ---------------------- STEP 1: DESCRIPTIVE STATISTICS ---------------------- ###

# Headline Length Analysis
df["headline_length"] = df["headline"].astype(str).apply(len)
print("Headline Length Stats:\n", df["headline_length"].describe())

plt.hist(df["headline_length"], bins=30, edgecolor='black')
plt.xlabel("Headline Length (characters)")
plt.ylabel("Frequency")
plt.title("Headline Length Distribution")
plt.show()

# Publisher Analysis
publisher_counts = df["publisher"].value_counts()
print("Articles Per Publisher:\n", publisher_counts)

plt.bar(publisher_counts.index[:10], publisher_counts[:10])
plt.xlabel("Publisher")
plt.ylabel("Number of Articles")
plt.title("Top 10 Publishers")
plt.xticks(rotation=45)
plt.show()

# Date Trend Analysis
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df.dropna(subset=["date"], inplace=True)
df["year_month"] = df["date"].dt.to_period("M")

date_counts = df["year_month"].value_counts().sort_index()
print("Date Trends:\n", date_counts)

plt.plot(date_counts.index.astype(str), date_counts.values, marker='o')
plt.xlabel("Year-Month")
plt.ylabel("Number of Articles")
plt.title("Publication Trends Over Time")
plt.grid(True)
plt.show()

### ---------------------- STEP 2: TEXT ANALYSIS (Topic Modeling with Financial Terms) ---------------------- ###

# Custom Financial Keyword Extraction
def extract_financial_keywords(text):
    text = text.lower()
    keywords = [term for term in financial_terms if term in text]
    return " ".join(keywords) if keywords else text  # Keep raw text if no keywords are found

df["filtered_headline"] = df["headline"].astype(str).apply(extract_financial_keywords)

# TF-IDF Vectorization (Focus on Financial Terms)
vectorizer = TfidfVectorizer(max_features=500)
tfidf_matrix = vectorizer.fit_transform(df["filtered_headline"])

# LDA Model with Optimized Parameters
lda_model = LatentDirichletAllocation(n_components=3, max_iter=5, verbose=1, random_state=42)
lda_model.fit(tfidf_matrix)

print(f"✅ Number of topics extracted: {lda_model.n_components}")

# Display Top Words per Topic
def display_topics(model, feature_names, num_words):
    print("✅ Entering topic display function...")  
    
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        print(f"✅ Topic {topic_idx + 1}: {', '.join(top_words)}")  

print("✅ Running Financial Topic Modeling Analysis...")
display_topics(lda_model, vectorizer.get_feature_names_out(), 10)

### ---------------------- STEP 3: TIME SERIES ANALYSIS (Publishing Trends) ---------------------- ###

# Extract Hourly & Daily Publishing Patterns
df["hour"] = df["date"].dt.hour
df["day"] = df["date"].dt.date

# Publishing Frequency Per Hour
hourly_counts = df["hour"].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.plot(hourly_counts.index, hourly_counts.values, marker="o")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Articles")
plt.title("Publishing Frequency Across Hours")
plt.grid()
plt.show()

# Publishing Frequency Over Time
daily_counts = df["day"].value_counts().sort_index()
plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, daily_counts.values, marker="o", linestyle="-")
plt.xlabel("Date")
plt.ylabel("Number of Articles")
plt.title("Daily Publishing Frequency Over Time")
plt.xticks(rotation=45)
plt.grid()
plt.show()

### ---------------------- STEP 4: PUBLISHER ANALYSIS ---------------------- ###

# Extract Domain Names from Publisher Emails (if applicable)
def extract_domain(email):
    if "@" in str(email):
        return email.split("@")[-1]  # Extract domain after "@"
    return "Unknown"

df["publisher_domain"] = df["publisher"].apply(extract_domain)

# Count Unique Domains
domain_counts = df["publisher_domain"].value_counts()
print("Publisher Domain Distribution:\n", domain_counts)

plt.figure(figsize=(10, 5))
domain_counts[:10].plot(kind="bar", color="skyblue")
plt.xlabel("Publisher Domain")
plt.ylabel("Number of Articles")
plt.title("Top 10 Publishing Domains")
plt.xticks(rotation=45)
plt.show()



 












