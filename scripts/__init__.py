# int.py

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Data Loading and Cleaning
# =========================

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean and preprocess data."""
    # Convert 'date' to datetime and create 'date' column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        raise KeyError("Column 'date' not found in DataFrame.")
    # Compute headline length if 'headline' exists
    if 'headline' in df.columns:
        df['headline_length'] = df['headline'].astype(str).str.len()
    else:
        df['headline_length'] = 0
    return df

# =========================
# Analysis Functions
# =========================

def plot_publication_trends(df):
    """Plot number of articles published over time."""
    if 'date' not in df.columns:
        raise KeyError("Column 'date' not found in DataFrame.")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.set_index('date')
    daily_counts = df.resample('D').size()
    plt.plot(daily_counts.index, daily_counts.values)
    plt.title('Articles Published Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.show()

def get_top_publishers(df, top_n=10):
    """Return top N publishers by article count."""
    if 'publisher' not in df.columns:
        raise KeyError("Column 'publisher' not found in DataFrame.")
    return df['publisher'].value_counts().head(top_n)

def plot_top_publishers(publisher_counts):
    """Plot bar chart of top publishers."""
    publisher_counts.plot(kind='bar')
    plt.title('Top Publishers')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.show()

# =========================
# Example Usage
# =========================

if __name__ == "__main__":
    # Replace 'your_data.csv' with your actual data file path
    data_path = r'c:\10 Kifia Tasks\data\raw_analyst_ratings.csv'
    
    # Load data
    df = load_data(data_path)
    
    # Clean data
    df = clean_data(df)
    
    # Plot publication trends over time
    plot_publication_trends(df)
    
    # Get and plot top publishers
    top_publishers = get_top_publishers(df)
    plot_top_publishers(top_publishers)