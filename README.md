# Financial-News-stock- Analysis Project

This project involves analyzing a dataset of market-related news articles to uncover insights through various exploratory and text analyses. The analysis includes descriptive statistics, publisher activity, temporal trends, topic modeling, and publisher contributions.

---

## Repository Setup & Workflow

- **Repository Creation:** A GitHub repository has been created to host all project code and documentation.
- **Branching Strategy:** A new branch named `task-1` has been created for this analysis.
- **Commit Frequency:** Work has been committed at least three times daily with descriptive messages.

---

## Data Analysis Overview

### 1. Descriptive Statistics
- Calculated basic statistics for text lengths (e.g., headline length) to understand size distribution.
- Counted the number of articles per publisher to identify the most active sources.
- Analyzed publication dates to discover news frequency trends over time, including peaks during significant events or specific days of the week.

### 2. Text Analysis (Topic Modeling)
- Utilized natural language processing techniques to extract common keywords and phrases.
- Identified probable topics and significant themes in the dataset such as "FDA approval," "price target," etc.

### 3. Time Series Analysis
- Investigated how article publication frequency varies over time.
- Detected spikes corresponding to specific market events.
- Analyzed publishing times to find patterns, such as peak times during the day when most news is released—useful for trading strategies.

### 4. Publisher Analysis
- Determined which publishers contribute most frequently to the news feed.
- Analyzed publisher differences to differentiate reporting styles or focus areas.
- If email addresses are used for publisher names, extracted and summarized unique domains to identify organizational contributors.

---

## Tools & Techniques Used
- Python (pandas, numpy, matplotlib, seaborn)
- Natural Language Processing (NLTK, spaCy)
- Topic modeling (LDA or similar)
- Time series analysis (statsmodels, pandas datetime)

---

## Conclusion
This analysis provides insights into news publication patterns, dominant sources, and key topics relevant for market participants and automated trading systems. It highlights the importance of timing and publisher influence in financial news dissemination.

---



## Task 2: Quantitative analysis using pynance and TaLib
Overview
This project focuses on quantitative stock price analysis and news sentiment correlation to understand financial market behavior. Using technical indicators and sentiment analysis, we aim to derive actionable insights into stock movements.

Objectives
Utilize technical indicators to analyze stock trends.

Apply sentiment analysis on financial news headlines.

Establish correlations between news sentiment and stock price movements.

Leverage PyNance & TA-Lib for financial data analysis.

Visualize findings using Matplotlib & Seaborn.

## Task 2: Quantitative Analysis using PyNance & TA-Lib
1️⃣ Load & Prepare Stock Data
Ensure the dataset includes Date, Open, High, Low, Close, Volume.

Convert the date column to a standardized format.

Set the date as the index to facilitate time-series analysis.

2️⃣ Apply Technical Indicators using TA-Lib
Install TA-Lib for financial analytics.

Compute Moving Average (SMA) to identify stock trends.

Calculate Relative Strength Index (RSI) to measure stock momentum.

Derive MACD (Moving Average Convergence Divergence) for trend analysis.

3️⃣ Financial Metrics with PyNance
Install PyNance for financial data processing.

Compute Bollinger Bands to analyze market volatility.

4️⃣ Visualization of Technical Indicators
Plot stock prices alongside technical indicators.

Use line graphs to illustrate SMA & RSI trends.


 ##  Task 3: Correlation between news and stock movement
     
1️⃣ Normalize Dates
Ensure both datasets align by normalizing timestamps.

Convert news article publication dates to match stock trading dates.

2️⃣ Perform Sentiment Analysis
Install NLTK for natural language processing.

Use VADER Sentiment Analysis to quantify the tone of financial news headlines.

3️⃣ Compute Daily Stock Returns
Calculate the daily percentage change in stock closing prices.

4️⃣ Merge Sentiment with Stock Data
Integrate sentiment scores with stock price movements by aligning dates.

5️⃣ Correlation Analysis
Measure the relationship between news sentiment and stock price fluctuations using statistical correlation methods.