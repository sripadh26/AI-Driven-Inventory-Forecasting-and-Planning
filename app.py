# app.py - Data Preprocessing for Retail Store Inventory Dataset
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

INPUT_CSV = "data/retail_store_inventory.csv"
TRAIN_OUT = "train_preprocessed.csv"
TEST_OUT = "test_preprocessed.csv"

print("="*60)
print("RETAIL STORE INVENTORY - DATA PREPROCESSING")
print("="*60)

# Load dataset
print(f"\n1. Loading dataset: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
print(f"   Original shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# Standardize column names
df.columns = df.columns.str.strip()

# Check for missing values
print(f"\n2. Missing values check:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("   No missing values found!")

# Convert Date column
print(f"\n3. Processing date column...")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values(['Product ID', 'Store ID', 'Date']).reset_index(drop=True)
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

# Extract time-based features
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Day_of_Month'] = df['Date'].dt.day
df['Week_of_Year'] = df['Date'].dt.isocalendar().week
df['Quarter'] = df['Date'].dt.quarter

# Weekend indicator
df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)

# Create lag features per product and store
print(f"\n4. Creating lag and rolling features...")
df = df.sort_values(['Product ID', 'Store ID', 'Date'])

# Lag features for Units Sold
for lag in [1, 3, 7, 14]:
    df[f'Units_Sold_Lag_{lag}'] = df.groupby(['Product ID', 'Store ID'])['Units Sold'].shift(lag)

# Rolling statistics
for window in [3, 7, 14]:
    df[f'Units_Sold_Rolling_Mean_{window}'] = (
        df.groupby(['Product ID', 'Store ID'])['Units Sold']
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )
    df[f'Units_Sold_Rolling_Std_{window}'] = (
        df.groupby(['Product ID', 'Store ID'])['Units Sold']
        .transform(lambda x: x.rolling(window=window, min_periods=1).std())
    )

# Demand-related features
df['Demand_Gap'] = df['Demand Forecast'] - df['Units Sold']
df['Inventory_to_Demand_Ratio'] = df['Inventory Level'] / (df['Demand Forecast'] + 1)
df['Price_to_Competitor_Ratio'] = df['Price'] / (df['Competitor Pricing'] + 0.01)

# Discount effectiveness
df['Discount_Rate'] = df['Discount'] / (df['Price'] + df['Discount'] + 0.01)
df['Effective_Price'] = df['Price'] - df['Discount']

# Stock indicators
df['Low_Stock'] = (df['Inventory Level'] < df['Demand Forecast']).astype(int)
df['Overstock'] = (df['Inventory Level'] > 2 * df['Demand Forecast']).astype(int)

# Promotional impact
df['Promo_With_Discount'] = df['Holiday/Promotion'] * (df['Discount'] > 0).astype(int)

# Fill NaN values created by lag features
print(f"\n5. Handling missing values from feature engineering...")
print(f"   Rows before cleaning: {len(df)}")

# Fill lag features with median per product-store group
lag_cols = [col for col in df.columns if 'Lag' in col or 'Rolling' in col]
for col in lag_cols:
    df[col] = df.groupby(['Product ID', 'Store ID'])[col].transform(
        lambda x: x.fillna(x.median())
    )

# Drop remaining NaN rows
df = df.dropna().reset_index(drop=True)
print(f"   Rows after cleaning: {len(df)}")

# Select final features for modeling
print(f"\n6. Selecting features for modeling...")
target = 'Units Sold'

base_features = [
    'Store ID', 'Product ID', 'Category', 'Region',
    'Inventory Level', 'Units Ordered', 'Demand Forecast',
    'Price', 'Discount', 'Weather Condition',
    'Holiday/Promotion', 'Competitor Pricing', 'Seasonality',
    'Epidemic'
]

time_features = [
    'Day_of_Week', 'Month', 'Quarter', 'Is_Weekend'
]

lag_features = [col for col in df.columns if 'Lag' in col or 'Rolling' in col]

engineered_features = [
    'Demand_Gap', 'Inventory_to_Demand_Ratio', 'Price_to_Competitor_Ratio',
    'Discount_Rate', 'Effective_Price', 'Low_Stock', 'Overstock',
    'Promo_With_Discount'
]

all_features = base_features + time_features + lag_features + engineered_features

# Ensure all features exist
all_features = [f for f in all_features if f in df.columns]

print(f"   Total features: {len(all_features)}")
print(f"   Feature categories:")
print(f"     - Base features: {len(base_features)}")
print(f"     - Time features: {len(time_features)}")
print(f"     - Lag features: {len([f for f in lag_features if f in df.columns])}")
print(f"     - Engineered features: {len([f for f in engineered_features if f in df.columns])}")

# Time-based train-test split (last 30 days for test)
print(f"\n7. Creating train-test split...")
last_date = df['Date'].max()
cutoff_date = last_date - pd.Timedelta(days=30)

train_df = df[df['Date'] <= cutoff_date].copy()
test_df = df[df['Date'] > cutoff_date].copy()

# Fallback to 80-20 split if test set too small
if len(test_df) < 100:
    print("   Test set too small, using 80-20 split instead...")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

print(f"   Train set: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Test set: {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")

# Save preprocessed data
print(f"\n8. Saving preprocessed data...")
train_df[all_features + [target]].to_csv(TRAIN_OUT, index=False)
test_df[all_features + [target]].to_csv(TEST_OUT, index=False)

print(f"   ✓ Saved: {TRAIN_OUT}")
print(f"   ✓ Saved: {TEST_OUT}")

# Summary statistics
print(f"\n9. Dataset Summary:")
print(f"   Target variable: {target}")
print(f"   Train {target} - Mean: {train_df[target].mean():.2f}, Std: {train_df[target].std():.2f}")
print(f"   Test {target} - Mean: {test_df[target].mean():.2f}, Std: {test_df[target].std():.2f}")
print(f"   Unique stores: {df['Store ID'].nunique()}")
print(f"   Unique products: {df['Product ID'].nunique()}")
print(f"   Unique categories: {df['Category'].nunique()}")

print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)