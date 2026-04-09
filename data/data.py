import pandas as pd 
import numpy as np

# --- PARAMS ---
INPUT_FILE = "data/online_retail.csv"
OUTPUT_X = "X.npy"
OUTPUT_Y = "Y.npy"

# --- LOADING ---
print("Loading Data ...")
df = pd.read_csv(INPUT_FILE, encoding="ISO-8859-1")
df.columns = df.columns.str.replace(" ", "")

df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors= 'coerce')

# --- CLEANING ---
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.dropna(subset=['CustomerID'])
df['CustomerID']=df['CustomerID'].astype(int)
df = df[df['Price']>=0]

df['is_return']= df['Invoice'].astype(str).str.upper().str.startswith('C')
df['LineValue']= df['Quantity']*df['Price']

global_last_date= df['InvoiceDate'].max()

# --- SPLIT PURCHASES\RETURNS ---
purchases = df[~df['is_return']].copy()
returns = df[df['is_return']].copy()

# --- DATA AGGREGATION ---

# --- Recency ---
last_purchase = (purchases.groupby('CustomerID')['InvoiceDate']
                 .max()
                 .rename('Last_Purchase_Date'))
recency = ((global_last_date - last_purchase).dt.days).rename('Recency')

# --- Frequency ---
frequency = (purchases.groupby('CustomerID')['Invoice']
             .nunique()
             .rename('Frequency'))

# --- Monetary Value ---
monetary = (purchases.groupby('CustomerID')['LineValue']
            .sum()
            .rename('Monetary_Value'))
log_monetary = np.log1p(monetary.clip(lower=0)).rename('Log_monetary_Value')

# --- Cancellation Rate ---
total_inv= df.groupby('CustomerID')['Invoice'].nunique().rename('Total_Inv')
cancel_inv = returns.groupby('CustomerID')['Invoice'].nunique().rename('Cancel_Inv')

cancel_rate = (cancel_inv.div(total_inv, fill_value = 0)
               .fillna(0)
               .rename('Cancellation_Rate'))

# --- StockCode Diversity ---
stock_div = (purchases.groupby('CustomerID')['StockCode']
             .nunique()
             .rename('StockCode_Diversity'))

# --- Return Propensity ---
qty_bought = (purchases.groupby('CustomerID')['Quantity']
              .apply(lambda x: x.clip(lower = 0).sum())
              .rename('Qty_Bought'))
qty_returned = (returns.groupby('CustomerID')['Quantity']
                .apply(lambda x: x.abs().sum())
                .rename("Qty_Returned"))

return_prop = (qty_returned.div(qty_bought.replace(0, np.nan))
               .fillna(0)
               .rename('Return_Propensity'))

# --- Avg Time Between Purchases (std dei gap in giorni) ---
def std_gap(dates):
    s = dates.sort_values()
    diffs = s.diff().dropna().dt.days
    return diffs.std() if len(diffs) >= 1 else 0.0

avg_time = (purchases.groupby("CustomerID")['InvoiceDate']
            .apply(std_gap)
            .rename('Avg_Time_Between_Purch')
            .fillna(0))

# --- Country dummy --- 
country_uk = (purchases.groupby('CustomerID')["Country"]
              .first()
              .str.strip().str.upper()
              .isin(["UNITED KINGDOM"])
              .astype(int)
              .rename('Country_UK'))

# --- CONCAT IN A SINGLE DATAFRAME ---
cs = pd.concat([
    recency, frequency, monetary, log_monetary,
    cancel_rate, stock_div, return_prop, 
    avg_time, country_uk, last_purchase], axis = 1).reset_index()

# --- CHURN VARIABLE ---
def mean_gap(dates):
    s = dates.dt.normalize().drop_duplicates().sort_values() # we drop duplicates for more orders in the same dat
    diffs = s.diff().dropna().dt.days
    return diffs.mean() if len(diffs) >= 1 else np.nan

mean_gap_per_customer = (purchases.groupby('CustomerID')['InvoiceDate']
                         .apply(mean_gap))

CHURN_WINDOW = int(mean_gap_per_customer.dropna().median())
churn_cutoff = global_last_date - pd.Timedelta(days=CHURN_WINDOW)

print(f"mean gap per customer: {mean_gap_per_customer.head()} days")
print(f"Churn Window (median value of the mean gaps): {CHURN_WINDOW} days")
print(f"Churn cutoff: {churn_cutoff}")

cs['Churn'] = (cs['Last_Purchase_Date'] < churn_cutoff).astype(int)

valid_customer = purchases['CustomerID'].unique()
cs = cs[cs['CustomerID'].isin(valid_customer)]

cs.to_excel('cleaned_data.xlsx', index = False)

# --- OUTPUT NUMPY ARRAY ---
FEATURE_COLS = ["Recency", "Frequency", "Log_monetary_Value",
    "Cancellation_Rate", "StockCode_Diversity", "Return_Propensity",
    "Avg_Time_Between_Purch", "Country_UK"]

X = cs[FEATURE_COLS].to_numpy(dtype=np.float64)
y = cs["Churn"].to_numpy(dtype=np.int32)

np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)

# -- RESULTS IN THE CHURN: IS THE DATASET BALANCED? --
n_churn = y.sum()
n_no_churn = len(y) - n_churn

print(f"Churn (1): {n_churn} ({n_churn/len(y)})")
print(f"Churn (0): {n_no_churn} ({n_no_churn/len(y)})")