# Financial News Analysis flow

This repository provides a comprehensive analysis pipeline for financial news articles. It covers:

- **Descriptive Statistics**: Key insights into text lengths, publisher activity, and publication trends over time.
- **Topic Modeling**: Identification of prevalent keywords and themes using natural language processing.
- **Time Series Analysis**: Examination of publication frequency over days and hours, helping identify market-sensitive news spikes or optimal times for news release.

---

## Features

- **Textual Length Analysis**: Calculate and visualize the distribution of headline lengths.
- **Publisher Activity**: Count and identify the most active news sources, including domain extraction from email addresses.
- **Publication Trends**: Analyze how news volume varies monthly and daily, and over specific hours.
- **Topic Extraction**: Use Latent Dirichlet Allocation (LDA) for uncovering prevalent topics and keywords from headlines.
- **Time-based Insights**: Detect spikes in news flow related to market events and identify the best times for news dissemination.

---

## Setup & Usage

### Prerequisites

- Python 3.x
- Essential libraries:
  - pandas
  - matplotlib
  - scikit-learn
  - nltk
  - re (built-in)
  - string (built-in)

### Installation
Install the required Python libraries:

```bash  
pip install pandas matplotlib scikit-learn nltk  