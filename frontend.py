import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

st.set_page_config(page_title="Inventory Demand Forecasting", layout="centered")

# ================= Load Models and Objects =================
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
lstm_model = load_model("lstm_model.keras", compile=False)
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# ================= Helpers =================
def safe_transform(le, values):
    classes = set(le.classes_)
    transformed = []
    for v in values:
        v = str(v)
        if v in classes:
            transformed.append(int(le.transform([v])[0]))
        else:
            transformed.append(0)
    return transformed

def ensure_numeric(df):
    numeric_cols = [
        "Inventory Level", "Units Ordered", "Demand Forecast",
        "Price", "Discount", "Competitor Pricing"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def encode_row_df(df_row):
    tmp = df_row.copy()
    for col, le in encoders.items():
        if col in tmp.columns:
            tmp[col] = safe_transform(le, tmp[col].astype(str))
    tmp = ensure_numeric(tmp)
    return tmp

def predict_row(df_row):
    encoded = encode_row_df(df_row)
    pred_xgb = xgb_model.predict(encoded)[0]
    scaled = scaler.transform(encoded)
    scaled = np.reshape(scaled, (scaled.shape[0], 1, scaled.shape[1]))
    pred_lstm = lstm_model.predict(scaled)[0][0]
    return float(pred_xgb), float(pred_lstm)

# ================= UI =================
st.title("📊 Inventory Demand Forecasting System")
st.write("Upload dataset (CSV/XLSX/XML) or enter a single record to predict demand and run price optimization.")

# ---------------- File upload ----------------
st.subheader("📂 Upload Inventory Dataset (CSV / XLSX / XML)")
uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xml"])

required_cols = ['Store ID', 'Product ID', 'Category', 'Region',
                 'Inventory Level', 'Units Ordered', 'Demand Forecast',
                 'Price', 'Discount', 'Weather Condition',
                 'Holiday/Promotion', 'Competitor Pricing', 'Seasonality']

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        new_data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        new_data = pd.read_excel(uploaded_file)
    else:
        new_data = pd.read_xml(uploaded_file)

    st.success("✅ File uploaded successfully!")
    st.dataframe(new_data.head())

    # Check required columns
    missing = [c for c in required_cols if c not in new_data.columns]
    if missing:
        st.error(f"Uploaded file is missing required columns: {missing}")
    else:
        # Encode categorical
        encoded_df = new_data.copy()
        for col, le in encoders.items():
            if col in encoded_df.columns:
                encoded_df[col] = encoded_df[col].astype(str).map(
                    lambda s: int(le.transform([s])[0]) if s in le.classes_ else 0
                )

        encoded_df = ensure_numeric(encoded_df)

        # Predict bulk
        preds_xgb = xgb_model.predict(encoded_df[required_cols])
        scaled_bulk = scaler.transform(encoded_df[required_cols])
        scaled_bulk = np.reshape(scaled_bulk, (scaled_bulk.shape[0], 1, scaled_bulk.shape[1]))
        preds_lstm = lstm_model.predict(scaled_bulk).flatten()

        new_data["Predicted_XGBoost"] = preds_xgb
        new_data["Predicted_LSTM"] = preds_lstm

        st.subheader("🧾 Bulk Predictions (first 10 rows)")
        st.dataframe(new_data.head(10))

        # Graph for uploaded data
        st.subheader("📈 Predicted Demand Overview")
        fig_all, ax_all = plt.subplots(figsize=(8, 4))
        ax_all.plot(new_data.index, new_data["Predicted_XGBoost"], label="XGBoost", marker='o')
        ax_all.plot(new_data.index, new_data["Predicted_LSTM"], label="LSTM", marker='x')
        ax_all.set_xlabel("Row index")
        ax_all.set_ylabel("Predicted Demand")
        ax_all.legend()
        st.pyplot(fig_all)

        # Row selection for Bayesian Optimization
        st.subheader("🔎 Select a row for optimization")
        options = [f"Row {i}" for i in new_data.index]
        selection = st.selectbox("Choose row", options)
        sel_idx = int(selection.split()[1])
        row = new_data.loc[[sel_idx]].copy().reset_index(drop=True)
        st.write("Selected row:")
        st.table(row.T)

        # Forecast
        st.subheader("📆 30-Day Forecast for Selected Row")
        pred_xgb_row = float(row["Predicted_XGBoost"].values[0])
        pred_lstm_row = float(row["Predicted_LSTM"].values[0])
        days = np.arange(1, 31)
        xgb_forecast = [pred_xgb_row]*30
        lstm_forecast = np.linspace(pred_lstm_row*0.95, pred_lstm_row*1.05, 30)

        fig_row, ax_row = plt.subplots(figsize=(8, 4))
        ax_row.plot(days, xgb_forecast, label="XGBoost", color="blue")
        ax_row.plot(days, lstm_forecast, label="LSTM", color="green")
        ax_row.set_xlabel("Day")
        ax_row.set_ylabel("Predicted Demand")
        ax_row.legend()
        st.pyplot(fig_row)

        # Bayesian Optimization for Price
        st.subheader("💰 Price Optimization for Selected Row")

        base_price = float(row["Price"].values[0])

        def objective_price(price_val):
            tmp = row.copy()
            tmp["Price"] = price_val
            tmp_enc = encode_row_df(tmp)
            tmp_enc = ensure_numeric(tmp_enc)
            pred = xgb_model.predict(tmp_enc[required_cols])[0]
            return float(pred)

        pbounds = {"price_val": (base_price*0.7, base_price*1.3)}
        optimizer = BayesianOptimization(f=objective_price, pbounds=pbounds, random_state=42, verbose=0)
        with st.spinner("Running Bayesian Optimization..."):
            optimizer.maximize(init_points=5, n_iter=10)

        opt_price = optimizer.max["params"]["price_val"]
        opt_sales = optimizer.max["target"]

        st.write(f"✅ Optimal Price: ₹{opt_price:.2f}")
        st.write(f"📈 Expected Sales at Optimal Price: {opt_sales:.2f} units")

        price_range = np.linspace(base_price*0.7, base_price*1.3, 20)
        sales_preds = [objective_price(p) for p in price_range]

        fig_price, ax_price = plt.subplots(figsize=(8, 4))
        ax_price.plot(price_range, sales_preds, marker='o', color='purple')
        ax_price.axvline(opt_price, color='red', linestyle='--', label=f"Optimal Price: {opt_price:.2f}")
        ax_price.set_xlabel("Price")
        ax_price.set_ylabel("Predicted Sales")
        ax_price.legend()
        st.pyplot(fig_price)

        csv = new_data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions as CSV", csv, "predictions.csv", "text/csv")

st.markdown("---")

# ---------------- Manual Input ----------------
st.subheader("🧮 Manual Input Prediction")

with st.form("single_form"):
    store_id = st.text_input("Store ID", "S001")
    product_id = st.text_input("Product ID", "P001")
    category = st.selectbox("Category", ["Electronics", "Clothing", "Groceries", "Furniture", "Cosmetics"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    inventory_level = st.number_input("Inventory Level", min_value=0, value=100)
    units_ordered = st.number_input("Units Ordered", min_value=0, value=50)
    demand_forecast = st.number_input("Demand Forecast", min_value=0, value=70)
    price = st.number_input("Price", min_value=0.0, value=20.0)
    discount = st.number_input("Discount", min_value=0.0, value=2.0)
    weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cold", "Hot"])
    holiday_promo = st.selectbox("Holiday/Promotion", ["Yes", "No"])
    competitor_price = st.number_input("Competitor Pricing", min_value=0.0, value=22.0)
    seasonality = st.selectbox("Seasonality", ["High", "Medium", "Low"])
    submitted_single = st.form_submit_button("Predict")

if submitted_single:
    row = pd.DataFrame([{
        "Store ID": store_id,
        "Product ID": product_id,
        "Category": category,
        "Region": region,
        "Inventory Level": inventory_level,
        "Units Ordered": units_ordered,
        "Demand Forecast": demand_forecast,
        "Price": price,
        "Discount": discount,
        "Weather Condition": weather,
        "Holiday/Promotion": holiday_promo,
        "Competitor Pricing": competitor_price,
        "Seasonality": seasonality
    }])
    pred_xgb, pred_lstm = predict_row(row)
    st.write(f"✅ XGBoost Prediction: {pred_xgb:.2f} units")
    st.write(f"✅ LSTM Prediction: {pred_lstm:.2f} units")

    days = np.arange(1, 31)
    fig_s, ax_s = plt.subplots(figsize=(8, 4))
    ax_s.plot(days, [pred_lstm]*30, label="LSTM Forecast")
    ax_s.plot(days, [pred_xgb]*30, label="XGBoost Forecast")
    ax_s.legend()
    st.pyplot(fig_s)
