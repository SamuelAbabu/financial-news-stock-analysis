import unittest
import pandas as pd
import numpy as np

# Sample Stock Data for Testing
df_stock = pd.DataFrame({
    "Date": pd.date_range(start="2025-01-01", periods=100, freq="D"),
    "Close": [100 + i * 0.5 for i in range(100)]
})
df_stock.set_index("Date", inplace=True)

class TestFinancialAnalysis(unittest.TestCase):

    # ✅ Test Moving Average Calculation
    def test_sma_calculation(self):
        df_stock["SMA_50"] = df_stock["Close"].rolling(window=50).mean()
        self.assertEqual(round(df_stock["SMA_50"].iloc[-1], 2), round(df_stock["Close"].iloc[-50:].mean(), 2))

    # ✅ Test RSI Calculation
    def test_rsi_calculation(self):
        df_stock["Price Change"] = df_stock["Close"].diff()
        df_stock["Gain"] = np.where(df_stock["Price Change"] > 0, df_stock["Price Change"], 0)
        df_stock["Loss"] = np.where(df_stock["Price Change"] < 0, abs(df_stock["Price Change"]), 0)
        df_stock["Avg Gain"] = df_stock["Gain"].rolling(window=14).mean()
        df_stock["Avg Loss"] = df_stock["Loss"].rolling(window=14).mean()
        df_stock["RS"] = df_stock["Avg Gain"] / df_stock["Avg Loss"]
        df_stock["RSI"] = 100 - (100 / (1 + df_stock["RS"]))
        self.assertTrue(df_stock["RSI"].max() <= 100)
        self.assertTrue(df_stock["RSI"].min() >= 0)

    # ✅ Test MACD Calculation
    def test_macd_calculation(self):
        df_stock["EMA_12"] = df_stock["Close"].ewm(span=12, adjust=False).mean()
        df_stock["EMA_26"] = df_stock["Close"].ewm(span=26, adjust=False).mean()
        df_stock["MACD"] = df_stock["EMA_12"] - df_stock["EMA_26"]
        df_stock["MACD_Signal"] = df_stock["MACD"].ewm(span=9, adjust=False).mean()
        self.assertTrue(isinstance(df_stock["MACD"].iloc[-1], float))
        self.assertTrue(isinstance(df_stock["MACD_Signal"].iloc[-1], float))

if __name__ == "__main__":
    unittest.main()
