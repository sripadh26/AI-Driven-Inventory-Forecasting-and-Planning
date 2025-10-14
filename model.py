# model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load datasets
train = pd.read_csv("train_preprocessed.csv")
test = pd.read_csv("test_preprocessed.csv")

# Target variable
target = "Units Sold"

# Features
features = ['Store ID', 'Product ID', 'Category', 'Region',
            'Inventory Level', 'Units Ordered', 'Demand Forecast',
            'Price', 'Discount', 'Weather Condition',
            'Holiday/Promotion', 'Competitor Pricing', 'Seasonality']

X_train = train[features].copy()
y_train = train[target].copy()
X_test = test[features].copy()
y_test = test[target].copy()

# ------------------- Encode Categorical -------------------
categorical_cols = ['Store ID', 'Product ID', 'Category', 'Region',
                    'Weather Condition', 'Seasonality', 'Holiday/Promotion']

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    encoders[col] = le

# Save encoders for frontend
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
# ------------------- XGBoost -------------------
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb) * 100
print(f"XGBoost Accuracy: {r2_xgb:.2f}%")

# Save XGBoost model
pickle.dump(xgb_model, open("xgb_model.pkl", "wb"))

# ------------------- LSTM -------------------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lstm_model = Sequential()
lstm_model.add(LSTM(64, activation="tanh", input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer="adam", loss="mean_squared_error")

lstm_model.fit(X_train_lstm, y_train, epochs=15, batch_size=32, verbose=1)

y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
r2_lstm = r2_score(y_test, y_pred_lstm) * 100
print(f"LSTM Accuracy: {r2_lstm:.2f}%")

# Save LSTM + Scaler
lstm_model.save("lstm_model.keras")   # modern Keras format
pickle.dump(scaler, open("scaler.pkl", "wb"))
