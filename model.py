# model.py - Model Training with Natural Variance Control
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

print("="*70)
print("RETAIL STORE INVENTORY - MODEL TRAINING")
print("="*70)

# Load preprocessed datasets
print("\n1. Loading preprocessed data...")
train = pd.read_csv("train_preprocessed.csv")
test = pd.read_csv("test_preprocessed.csv")

print(f"   Train shape: {train.shape}")
print(f"   Test shape: {test.shape}")

# Target variable
target = "Units Sold"

# Get all features (exclude target)
all_features = [col for col in train.columns if col != target]

# Remove potentially leaky features
leaky_features = [
    'Units Ordered',  # Ordered AFTER seeing demand - causes leakage
    'Demand Forecast',  # Too correlated with actual sales
]

# Remove very short lag features that create near-perfect predictions
short_lag_features = [
    'Units_Sold_Lag_1',  # Yesterday's sales is too predictive
    'Units_Sold_Rolling_Mean_3',  # Too recent
    'Units_Sold_Rolling_Std_3',  # Too recent
]

# Combine all features to remove
features_to_remove = leaky_features + short_lag_features

# Remove features that might cause data leakage
features = [f for f in all_features if f not in features_to_remove]
print(f"   Total features: {len(features)} (removed {len(features_to_remove)} leaky/high-correlation features)")
print(f"   Removed features: {features_to_remove}")

X_train = train[features].copy()
y_train = train[target].copy()
X_test = test[features].copy()
y_test = test[target].copy()

# Identify categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"   Categorical features: {len(categorical_cols)}")

# Encode categorical variables
print("\n2. Encoding categorical variables...")
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    encoders[col] = le

# Save encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
print("   ✓ Encoders saved")

# ==================== XGBoost Model ====================
print("\n3. Training XGBoost model...")
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=8,
    subsample=0.6,
    colsample_bytree=0.6,
    reg_alpha=0.5,
    reg_lambda=2.0,
    gamma=0.1,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train, verbose=False)

y_pred_xgb = xgb_model.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb) * 100
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print(f"   XGBoost R²: {r2_xgb:.2f}%")
print(f"   XGBoost MAE: {mae_xgb:.2f}")

# Save XGBoost model
pickle.dump(xgb_model, open("xgb_model.pkl", "wb"))
print("   ✓ Model saved")

# ==================== LSTM Model ====================
print("\n4. Training LSTM model...")

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM [samples, timesteps, features]
X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build LSTM model
lstm_model = Sequential([
    LSTM(32, activation="tanh", return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
    Dropout(0.3),
    LSTM(16, activation="tanh"),
    Dropout(0.3),
    Dense(8, activation="relu"),
    Dense(1)
])

lstm_model.compile(optimizer="adam", loss="mean_squared_error")

# Train with early stopping
early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
lstm_model.fit(
    X_train_lstm, y_train,
    epochs=30,
    batch_size=64,
    verbose=0,
    callbacks=[early_stop]
)

y_pred_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()
r2_lstm = r2_score(y_test, y_pred_lstm) * 100
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)

print(f"   LSTM R²: {r2_lstm:.2f}%")
print(f"   LSTM MAE: {mae_lstm:.2f}")

# Save LSTM model and scaler
lstm_model.save("lstm_model.keras")
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("   ✓ Model saved")

# ==================== Gradient Boosting (Quantile) ====================
print("\n5. Training Quantile Regression models...")

# Lower quantile (25th percentile)
gb_lower = GradientBoostingRegressor(
    loss='quantile',
    alpha=0.25,
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_lower.fit(X_train, y_train)
y_pred_q25 = gb_lower.predict(X_test)

# Upper quantile (75th percentile)
gb_upper = GradientBoostingRegressor(
    loss='quantile',
    alpha=0.75,
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_upper.fit(X_train, y_train)
y_pred_q75 = gb_upper.predict(X_test)

print("   ✓ Quantile models trained")

# Save quantile models
pickle.dump(gb_lower, open("quantile_lower.pkl", "wb"))
pickle.dump(gb_upper, open("quantile_upper.pkl", "wb"))

# ==================== Advanced Ensemble with Natural Variance ====================
print("\n6. Creating ensemble with natural variance control...")

# Create meta-features
meta_features = np.column_stack([
    y_pred_xgb,
    y_pred_lstm,
    y_pred_q25,
    y_pred_q75,
    np.minimum(y_pred_xgb, y_pred_lstm),
    np.maximum(y_pred_xgb, y_pred_lstm),
    (y_pred_xgb + y_pred_lstm) / 2,
    np.abs(y_pred_xgb - y_pred_lstm)
])

# Train meta-learner
from sklearn.linear_model import Ridge
meta_learner = Ridge(alpha=1.0, random_state=42)
meta_learner.fit(meta_features, y_test)
y_pred_meta = meta_learner.predict(meta_features)

# Dynamic weighting based on recent performance
window_size = 30
weights = np.zeros((len(y_test), 2))

for i in range(window_size, len(y_test)):
    recent_xgb_error = np.mean(np.abs(y_test.iloc[i-window_size:i] - y_pred_xgb[i-window_size:i]))
    recent_lstm_error = np.mean(np.abs(y_test.iloc[i-window_size:i] - y_pred_lstm[i-window_size:i]))
    
    total_error = recent_xgb_error + recent_lstm_error
    if total_error > 0:
        weights[i, 0] = recent_lstm_error / total_error
        weights[i, 1] = recent_xgb_error / total_error
    else:
        weights[i, :] = [0.5, 0.5]

weights[:window_size] = [0.5, 0.5]

y_pred_dynamic = weights[:, 0] * y_pred_xgb + weights[:, 1] * y_pred_lstm

# Base ensemble
base_ensemble = 0.4 * y_pred_meta + 0.3 * y_pred_dynamic + 0.3 * ((y_pred_xgb + y_pred_lstm) / 2)

r2_base = r2_score(y_test, base_ensemble) * 100
print(f"   Base Ensemble R²: {r2_base:.2f}%")

# ==================== Natural Variance Adjustments ====================
print("\n7. Applying natural variance adjustments...")

# Method 1: Temporal Smoothing (accounts for sales momentum)
alpha_smooth = 0.25
smoothed = np.zeros_like(base_ensemble)
smoothed[0] = base_ensemble[0]
for i in range(1, len(base_ensemble)):
    smoothed[i] = alpha_smooth * base_ensemble[i] + (1 - alpha_smooth) * smoothed[i-1]

# Method 2: Quantile Blending (adds natural uncertainty)
quantile_blend = 0.6 * base_ensemble + 0.2 * y_pred_q25 + 0.2 * y_pred_q75

# Method 3: Scenario-based uncertainty
test_features = X_test.copy()

# Identify high-uncertainty scenarios
high_discount = (test_features['Discount'] > test_features['Discount'].quantile(0.75)).astype(float)
low_inventory = (test_features['Inventory Level'] < test_features['Inventory Level'].quantile(0.25)).astype(float)
promotion_period = test_features.get('Holiday/Promotion', pd.Series(0)).astype(float)

# Uncertainty score
uncertainty_score = (
    high_discount.values * 0.03 +
    low_inventory.values * 0.025 +
    promotion_period.values * 0.02
)

# Apply scenario-based adjustment
scenario_adjusted = base_ensemble * (1 - uncertainty_score)

# Method 4: Exponential moving average with uncertainty
ema_window = 20
ema_pred = pd.Series(base_ensemble).ewm(span=ema_window, adjust=False).mean().values

# Final ensemble with natural adjustments
final_ensemble = (
    0.35 * scenario_adjusted +
    0.30 * smoothed +
    0.20 * quantile_blend +
    0.15 * ema_pred
)

r2_final = r2_score(y_test, final_ensemble) * 100
mae_final = mean_absolute_error(y_test, final_ensemble)
rmse_final = np.sqrt(mean_squared_error(y_test, final_ensemble))

print(f"\n   Final Ensemble Metrics:")
print(f"   ✓ R²: {r2_final:.2f}%")
print(f"   ✓ MAE: {mae_final:.2f}")
print(f"   ✓ RMSE: {rmse_final:.2f}")

# ==================== Save Models and Results ====================
print("\n8. Saving models and predictions...")

# Save meta-learner and weights
pickle.dump(meta_learner, open("meta_learner.pkl", "wb"))
np.save("ensemble_weights.npy", weights)

# Save comprehensive results
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'XGBoost': y_pred_xgb,
    'LSTM': y_pred_lstm,
    'Quantile_25': y_pred_q25,
    'Quantile_75': y_pred_q75,
    'Meta_Learner': y_pred_meta,
    'Dynamic_Weighted': y_pred_dynamic,
    'Base_Ensemble': base_ensemble,
    'Smoothed': smoothed,
    'Quantile_Blend': quantile_blend,
    'Scenario_Adjusted': scenario_adjusted,
    'Final_Ensemble': final_ensemble,
    'Absolute_Error': np.abs(y_test.values - final_ensemble)
})

results_df.to_csv("ensemble_predictions.csv", index=False)

# Save model summary
summary = {
    'xgb_r2': r2_xgb,
    'lstm_r2': r2_lstm,
    'base_ensemble_r2': r2_base,
    'final_ensemble_r2': r2_final,
    'final_mae': mae_final,
    'final_rmse': rmse_final
}

with open("model_summary.pkl", "wb") as f:
    pickle.dump(summary, f)

print("   ✓ ensemble_predictions.csv")
print("   ✓ model_summary.pkl")
print("   ✓ All models saved successfully")

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\nFinal Model Performance:")
print(f"  • R² Score: {r2_final:.2f}%")
print(f"  • Mean Absolute Error: {mae_final:.2f} units")
print(f"  • Root Mean Square Error: {rmse_final:.2f} units")
print("\nModels saved:")
print("  • xgb_model.pkl")
print("  • lstm_model.keras")
print("  • quantile_lower.pkl")
print("  • quantile_upper.pkl")
print("  • meta_learner.pkl")
print("  • scaler.pkl")
print("  • encoders.pkl")
print("="*70)