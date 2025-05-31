import pandas as pd
import matplotlib.pyplot as plt
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords

# ------------------ Settings ------------------
# Path to CSV data file
DATA_PATH = 'C:/10 Kifia Tasks/data/raw_analyst_ratings.csv'

# ------------------ Load Data ------------------
df = pd.read_csv(DATA_PATH)

# ------------------ Descriptive Statistics ------------------

# Headline length stats
df["headline_length"] = df["headline"].astype(str).apply(len)
print("Headline Length Statistics:\n", df["headline_length"].describe())

# How many articles per publisher
publisher_counts = df["publisher"].value_counts()
print("\nArticles per Publisher:\n", publisher_counts)

# Extract domain names from publisher field (if email addresses are present)
def extract_domain(publisher):
    # If email address
    if '@' in str(publisher):
        return publisher.split('@')[-1].split('.')[0]
    # Else assume it's a domain
    else:
        return str(publisher).split('.')[0]

df['publisher_domain'] = df['publisher'].apply(extract_domain)
domain_counts = df['publisher_domain'].value_counts()
print("\nArticles per Publisher Domain:\n", domain_counts)

# Analyze date and publication trends
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df.dropna(subset=["date"], inplace=True)
df["year_month"] = df["date"].dt.to_period("M")
monthly_counts = df["year_month"].value_counts().sort_index()

print("\nPublication Trends Over Time:\n", monthly_counts)

# Plot publication over time
plt.figure(figsize=(12, 6))
monthly_counts.plot(kind='line', marker='o')
plt.xlabel("Year-Month")
plt.ylabel("Number of Articles")
plt.title("Monthly Publication Trends")
plt.show()

# ------------------ Text Preprocessing for Topic Modeling ------------------

# Initialize stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# Remove punctuation using regex for speed
punct_regex = re.compile(r'[{}]'.format(re.escape(string.punctuation)))

def preprocess_text(text):
    text = str(text).lower()
    text = punct_regex.sub("", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["clean_headline"] = df["headline"].apply(preprocess_text)

# ------------------ Topic Modeling ------------------

vectorizer = TfidfVectorizer(max_features=1000, n_jobs=-1)
tfidf_matrix = vectorizer.fit_transform(df["clean_headline"])

lda = LatentDirichletAllocation(n_components=5, random_state=42, n_jobs=-1)
lda.fit(tfidf_matrix)

# Function to display topics
def display_topics(model, feature_names, top_n=10):
    for idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-top_n - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"\nTopic {idx + 1}: {', '.join(top_words)}")

display_topics(lda, vectorizer.get_feature_names_out(), 10)

# ------------------ Time Series Analysis ------------------

# Articles count per day
daily_counts = df["date"].dt.date.value_counts().sort_index()

# Plot daily publication frequency
plt.figure(figsize=(14,6))
daily_counts.plot()
plt.xlabel("Date")
plt.ylabel("Number of Articles")
plt.title("Daily News Articles Volume")
plt.show()

# ------------------ Publishing Time Analysis ------------------

# Extract hour from datetime for publication time analysis
df['hour'] = df['date'].dt.hour
hour_counts = df['hour'].value_counts().sort_index()

# Plot distribution of publishing hours
plt.figure(figsize=(10,6))
hour_counts.plot(kind='bar')
plt.xlabel("Hour of Day")
plt.ylabel("Number of Articles")
plt.title("Articles Published by Hour of Day")
plt.show()

# ------------------ Summary and Insights ------------------
print(f"\nMost active publishers:\n{publisher_counts.head(10)}")
print(f"\nMost frequent publisher domains:\n{domain_counts.head(10)}")
print(f"\nTop topics:\n")
display_topics(lda, vectorizer.get_feature_names_out(), 10)

# ------------------ Stock Price Analysis ------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ✅ Load Multiple Stock Price Datasets
csv_files = [
    r"c:\10 Kifia Tasks\yfinance_data\AAPL_historical_data.csv",
    r"c:\10 Kifia Tasks\yfinance_data\AMZN_historical_data.csv",
    r"c:\10 Kifia Tasks\yfinance_data\GOOG_historical_data.csv",
    r"c:\10 Kifia Tasks\yfinance_data\META_historical_data.csv",
    r"c:\10 Kifia Tasks\yfinance_data\MSFT_historical_data.csv",
    r"c:\10 Kifia Tasks\yfinance_data\NVDA_historical_data.csv",
    r"c:\10 Kifia Tasks\yfinance_data\TSLA_historical_data.csv"
]

stock_data = {}
for file in csv_files:
    stock_name = file.split("\\")[-1].split("_")[0]  # Extract stock ticker
    stock_data[stock_name] = pd.read_csv(file, parse_dates=["Date"], index_col="Date")

print(stock_data["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA"]:.head())
print("🚀 Stock datasets loaded successfully!")

# ✅ Compute Financial Metrics Without TA-Lib
for stock in stock_data:
    stock_data[stock]["SMA_50"] = stock_data[stock]["Close"].rolling(window=50).mean()
    stock_data[stock]["SMA_200"] = stock_data[stock]["Close"].rolling(window=200).mean()

    stock_data[stock]["Price Change"] = stock_data[stock]["Close"].diff()
    stock_data[stock]["Gain"] = np.where(stock_data[stock]["Price Change"] > 0, stock_data[stock]["Price Change"], 0)
    stock_data[stock]["Loss"] = np.where(stock_data[stock]["Price Change"] < 0, abs(stock_data[stock]["Price Change"]), 0)
    stock_data[stock]["Avg Gain"] = stock_data[stock]["Gain"].rolling(window=14).mean()
    stock_data[stock]["Avg Loss"] = stock_data[stock]["Loss"].rolling(window=14).mean()
    stock_data[stock]["RS"] = stock_data[stock]["Avg Gain"] / stock_data[stock]["Avg Loss"]
    stock_data[stock]["RSI"] = 100 - (100 / (1 + stock_data[stock]["RS"]))

    stock_data[stock]["EMA_12"] = stock_data[stock]["Close"].ewm(span=12, adjust=False).mean()
    stock_data[stock]["EMA_26"] = stock_data[stock]["Close"].ewm(span=26, adjust=False).mean()
    stock_data[stock]["MACD"] = stock_data[stock]["EMA_12"] - stock_data[stock]["EMA_26"]
    stock_data[stock]["MACD_Signal"] = stock_data[stock]["MACD"].ewm(span=9, adjust=False).mean()

print("🚀 Financial indicators computed successfully!")

# ✅ Visualize Stock Trends for Each Stock
stocks = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA"]

for stock in stocks:
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data[stock].index, stock_data[stock]["Close"], label="Stock Close Price", color="black", linewidth=2)
    plt.plot(stock_data[stock].index, stock_data[stock]["SMA_50"], label="50-Day SMA", color="blue", linestyle="--", linewidth=2)
    plt.plot(stock_data[stock].index, stock_data[stock]["SMA_200"], label="200-Day SMA", color="red", linestyle=":", linewidth=2)

    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Stock Price", fontsize=12)
    plt.title(f"{stock} Stock Price & Indicators", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()



