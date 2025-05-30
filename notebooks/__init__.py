import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 🔹 Load dataset
print("🔹 Loading Dataset")
df = pd.read_csv(r'c:\10 Kifia Tasks\data\raw_analyst_ratings.csv')

# 🔹 Convert Date Column for Analysis (with improved error handling)
df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d %H:%M:%S')

# 🔹 Check for NaT values and warn if necessary
missing_dates = df[df['date'].isna()]
if not missing_dates.empty:
    print("\n⚠️ Warning: Some rows contain invalid date formats. Review these:")
    print(missing_dates.head())

# 🔹 Headline Length Statistics
print("\n🔹 Headline Length Statistics")
df['headline_length'] = df['headline'].astype(str).str.len()
print(df['headline_length'].describe())

# 🔹 Count Articles Per Publisher (Corrected from 'publication' to 'publisher')
print("\n🔹 Top Publishers")
publisher_counts = df['publisher'].value_counts()
print(publisher_counts.head(10))

# 🔹 Analyze Publication Trends Over Time
print("\n🔹 Daily Publication Counts")
daily_counts = df.resample('D', on='date').size()
print(daily_counts.head())

# 🔹 Visualization of Publication Trends
print("\n🔹 Generating Visualization")

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12,6))

# Plot Publication Trends with a Trend Line
daily_counts.plot(ax=ax, color='royalblue', linewidth=2, label="Daily Counts")
daily_counts.rolling(window=7).mean().plot(ax=ax, linestyle="--", linewidth=2, color="darkred", label="7-day Avg")

# Formatting Graph
ax.set_title('Articles Published Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.show()



