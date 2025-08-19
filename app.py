
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ============ STEP 1: Load Dataset ============
df = pd.read_csv("data/retail_store_inventory.csv")
print("✅ Dataset loaded successfully")

# ============ STEP 2: Dataset Overview ============
print("\n📊 Dataset Info:")
print("Shape:", df.shape)
print("Columns:", df.columns)
print("\nMissing values:\n", df.isnull().sum())
print("\nPreview:\n", df.head())

# ============ STEP 3: EDA (Exploratory Data Analysis) ============

# Convert to datetime if date exists
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Sales trend
    if 'sales' in df.columns:
        df['sales'].resample('W').sum().plot(figsize=(10, 5), title="Weekly Sales Trend")
        plt.show()

# Top-selling products
if 'product_id' in df.columns and 'sales' in df.columns:
    top_products = df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_products.index, y=top_products.values)
    plt.title("Top 10 Best-Selling Products")
    plt.xlabel("Product ID")
    plt.ylabel("Total Sales")
    plt.show()

# Correlation heatmap (numeric features only)
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Reset index for preprocessing
df.reset_index(inplace=True)

# ============ STEP 4: Preprocessing ============

# Fill missing sales
if 'sales' in df.columns:
    df['sales'] = df['sales'].fillna(method='ffill')

# Fill missing price
if 'price' in df.columns:
    df['price'] = df['price'].fillna(df['price'].median())

# Feature Engineering
if 'sales' in df.columns:
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['rolling_7'] = df['sales'].rolling(window=7).mean()

# Time features
if 'date' in df.columns:
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

# Promotion flag
if 'price' in df.columns:
    avg_price = df['price'].mean()
    df['promotion'] = (df['price'] < avg_price).astype(int)

# Encoding categorical product_id
if 'product_id' in df.columns:
    encoder = LabelEncoder()
    df['product_id'] = encoder.fit_transform(df['product_id'])

# Scaling sales
if 'sales' in df.columns:
    scaler = MinMaxScaler()
    df['sales_scaled'] = scaler.fit_transform(df[['sales']])

# Drop NaN rows created by lag/rolling
df = df.dropna()

# ============ STEP 5: Train-Test Split ============
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

print("\n✅ Data split completed")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Save datasets
train.to_csv("train_preprocessed.csv", index=False)
test.to_csv("test_preprocessed.csv", index=False)

print("\n🎉 Preprocessing complete! Files saved as 'train_preprocessed.csv' and 'test_preprocessed.csv'")
