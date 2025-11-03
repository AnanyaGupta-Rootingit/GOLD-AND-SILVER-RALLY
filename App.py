import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# --- Streamlit Config ---
st.set_page_config(page_title="Gold & Silver Analysis Dashboard", layout="wide")

# --- Load and Prepare Data ---
@st.cache_data(show_spinner=False)
def load_data(gold_silver_path, macro_path):
    gold_silver = pd.read_excel(gold_silver_path)
    macro = pd.read_excel(macro_path)

    gold_silver['Date'] = pd.to_datetime(gold_silver['Date'])
    macro['Date'] = pd.to_datetime(macro['Date'])

    df = pd.merge(gold_silver, macro, on='Date', how='inner')
    df = df.sort_values('Date').reset_index(drop=True)

    # Standardize column names
    df.rename(columns={
        'Gold_Price': 'Gold_Close',
        'Silver_Price': 'Silver_Close'
    }, inplace=True)

    numeric_cols = ['Gold_Returns', 'Silver_Returns', 'DXY', 'US10Y_Yield']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Derived columns
    df['Gold_Volatility'] = df['Gold_Returns'].rolling(30).std() * (252**0.5)
    df['Silver_Volatility'] = df['Silver_Returns'].rolling(30).std() * (252**0.5)
    df['Gold_to_DXY_Corr'] = df['Gold_Returns'].rolling(90).corr(df['DXY'])
    df['Gold_to_Yield_Corr'] = df['Gold_Returns'].rolling(90).corr(df['US10Y_Yield'])
    df['Gold_Silver_Ratio'] = df['Gold_Close'] / df['Silver_Close']

    return df

# --- Rolling Regression Function ---
@st.cache_data(show_spinner=False)
def calculate_rolling_betas(df, window=180):
    df = df.dropna(subset=['Gold_Returns', 'India_CPI', 'DXY', 'US10Y_Yield', 'GoogleTrends_Gold'])
    rolling_results = []

    for i in range(window, len(df)):
        y = df['Gold_Returns'].iloc[i-window:i]
        X = df[['India_CPI', 'DXY', 'US10Y_Yield', 'GoogleTrends_Gold']].iloc[i-window:i]
        X = sm.add_constant(X)

        if X.isnull().values.any() or y.isnull().values.any():
            continue

        try:
            model = sm.OLS(y, X).fit()
            rolling_results.append({
                'Date': df['Date'].iloc[i],
                'Inflation_Beta': model.params.get('India_CPI', None),
                'DXY_Beta': model.params.get('DXY', None),
                'US10Y_Yield_Beta': model.params.get('US10Y_Yield', None),
                'Sentiment_Beta': model.params.get('GoogleTrends_Gold', None)
            })
        except:
            continue

    return pd.DataFrame(rolling_results)

# --- Streamlit UI ---
st.title("The Digital Bullion Rush — Decoding the 2025 Gold & Silver Rally")
st.write("""
This interactive dashboard explores how macroeconomic indicators, sentiment data, and precious metal dynamics 
influenced the 2025 gold and silver rally.
""")

# --- Load Data Section ---
st.header("📂 Data Loading")
st.write("Upload or ensure both Excel files exist in the working directory:")

uploaded_gold = st.file_uploader("Upload gold_silver_data.xlsx", type=["xlsx"])
uploaded_macro = st.file_uploader("Upload macro_data.xlsx", type=["xlsx"])

if uploaded_gold and uploaded_macro:
    df = load_data(uploaded_gold, uploaded_macro)
    st.success("Data loaded successfully!")
else:
    st.warning("Please upload both files to continue.")
    st.stop()

# --- Raw Data Preview ---
st.header("🔍 Raw Data Preview")
st.dataframe(df.head())

# --- Visualizations ---
st.header("📈 Key Visualizations")

## Gold & Silver Price Trend
st.subheader("Gold & Silver Price Rally (2020–2025)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['Gold_Close'], label='Gold Price', color='gold')
ax.plot(df['Date'], df['Silver_Close'], label='Silver Price', color='silver')
ax.set_title("Gold & Silver Price Rally (2020–2025)")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
plt.close(fig)

## Rolling Correlation: Gold vs DXY
st.subheader("Rolling Correlation: Gold vs DXY")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['Gold_to_DXY_Corr'], color='blue')
ax.set_title("Rolling Correlation between Gold and DXY (2020–2025)")
ax.set_xlabel("Date")
ax.set_ylabel("Rolling Correlation")
ax.grid(True)
st.pyplot(fig)
plt.close(fig)

## Rolling Correlation: Gold vs Yield
st.subheader("Rolling Correlation: Gold vs US 10Y Yield")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['Gold_to_Yield_Corr'], color='green')
ax.set_title("Rolling Correlation between Gold and 10Y Yield (2020–2025)")
ax.set_xlabel("Date")
ax.set_ylabel("Rolling Correlation")
ax.grid(True)
st.pyplot(fig)
plt.close(fig)

## Correlation Heatmap
st.subheader("Correlation Heatmap: Gold, Silver & Macros")
corr_cols = [col for col in ['Gold_Returns', 'Silver_Returns', 'India_CPI', 'DXY', 'US10Y_Yield', 
                             'GoogleTrends_Gold', 'GoogleTrends_Silver'] if col in df.columns]
corr_matrix = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", ax=ax)
ax.set_title("Correlation Heatmap: Gold, Silver & Macro Variables")
st.pyplot(fig)
plt.close(fig)

## Gold–Silver Ratio
st.subheader("Gold–Silver Ratio (2020–2025)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['Gold_Silver_Ratio'], color='purple')
ax.set_title("Gold–Silver Ratio (2020–2025)")
ax.set_xlabel("Date")
ax.set_ylabel("Ratio (Gold / Silver)")
ax.grid(True)
st.pyplot(fig)
plt.close(fig)

## Rolling Regression Coefficients
st.subheader("Rolling Regression Coefficients (Gold Returns vs Macros)")
rolling_df = calculate_rolling_betas(df)
if not rolling_df.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rolling_df['Date'], rolling_df['Inflation_Beta'], label='Inflation Beta', color='red')
    ax.plot(rolling_df['Date'], rolling_df['DXY_Beta'], label='DXY Beta', color='blue')
    ax.plot(rolling_df['Date'], rolling_df['US10Y_Yield_Beta'], label='Yield Beta', color='green')
    ax.plot(rolling_df['Date'], rolling_df['Sentiment_Beta'], label='Sentiment Beta', color='orange')
    ax.set_title("Rolling Regression Coefficients (Gold Returns vs Macro Drivers)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Coefficient Value")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("Could not calculate rolling regression betas. Possibly due to missing data.")

# --- Next Steps ---
st.header("🧠 Further Analysis Ideas")
st.write("""
- Add Prophet-based gold price forecast panel.  
- Include sentiment trend plots for Silver.  
- Add date range filters for dynamic chart updates.  
- Integrate ETF inflow data (GLD/SLV) for investor trend analysis.  
""")

st.info("✅ To run locally: `streamlit run dashboard_app.py`")
