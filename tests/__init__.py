import unittest
import pandas as pd
import re
import string

# Assuming the main functions are in a script named 'analysis_script.py'. For illustration, defining inline functions:

# Function to extract domain
def extract_domain(publisher):
    if '@' in str(publisher):
        return publisher.split('@')[-1].split('.')[0]
    else:
        return str(publisher).split('.')[0]

# Function to preprocess text
punct_regex = re.compile(r'[{}]'.format(re.escape(string.punctuation)))
def preprocess_text(text):
    text = str(text).lower()
    text = punct_regex.sub("", text)
    tokens = text.split()
    # Use a fixed stopword list for testing
    stop_words = set(["the", "is", "at", "which", "on"])
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

class TestAnalysisFunctions(unittest.TestCase):

    def test_extract_domain_email(self):
        email = "john.doe@finance.com"
        domain = extract_domain(email)
        self.assertEqual(domain, "finance")
    
    def test_extract_domain_domain_name(self):
        domain_name = "example.com"
        domain = extract_domain(domain_name)
        self.assertEqual(domain, "example")
    
    def test_preprocess_text_lowercase(self):
        raw_text = "The Price TARGET is high!"
        processed = preprocess_text(raw_text)
        self.assertIn("price", processed)
        self.assertIn("target", processed)
    
    def test_preprocess_text_removes_punctuation(self):
        raw_text = "FDA approval, announcement!"
        processed = preprocess_text(raw_text)
        self.assertNotIn(",", processed)
        self.assertNotIn("!", processed)
    
    def test_descriptive_statistics(self):
        # Mock dataframe
        data = {
            "headline": ["Market crashes", "Stocks rise", "Economic outlook"],
        }
        df_mock = pd.DataFrame(data)
        df_mock["headline_length"] = df_mock["headline"].astype(str).apply(len)
        desc = df_mock["headline_length"].describe()
        self.assertGreaterEqual(desc['mean'], 10)
        self.assertEqual(len(df_mock), 3)
    
    def test_monthly_counts(self):
        # Create sample dates
        date_series = pd.to_datetime(["2023-01-01", "2023-01-15", "2023-02-01"])
        df_test = pd.DataFrame({"date": date_series})
        df_test["year_month"] = df_test["date"].dt.to_period("M")
        counts = df_test["year_month"].value_counts()
        self.assertEqual(counts[0], 2)  # January count
        self.assertEqual(counts[1], 1)  # February count

if __name__ == '__main__':
    unittest.main()