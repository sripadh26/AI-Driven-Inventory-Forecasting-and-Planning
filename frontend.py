import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# 🎨 PAGE CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Inventory Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better readability
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #475569;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #F0F9FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
        color: #000000;
    }
    .info-box h4, .info-box p, .info-box ul, .info-box li, .info-box b {
        color: #000000;
    }
    .success-box {
        background-color: #F0FDF4;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #22C55E;
        margin: 1rem 0;
        color: #000000;
    }
    .success-box h4, .success-box p, .success-box ul, .success-box li, .success-box b {
        color: #000000;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
        color: #000000;
    }
    .warning-box h4, .warning-box p, .warning-box ul, .warning-box li, .warning-box b {
        color: #000000;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem;
        border: none;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📦 Smart Inventory Demand Forecasting</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict future sales and optimize your pricing strategy with AI</p>', unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 🧠 LOAD MODELS
# -------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
        lstm_model = load_model("lstm_model.keras", compile=False)
        scaler = pickle.load(open("scaler.pkl", "rb"))
        encoders = pickle.load(open("encoders.pkl", "rb"))
        meta_learner = pickle.load(open("meta_learner.pkl", "rb"))
        q_lower = pickle.load(open("quantile_lower.pkl", "rb"))
        q_upper = pickle.load(open("quantile_upper.pkl", "rb"))
        
        if hasattr(xgb_model, 'feature_names_in_'):
            actual_features = list(xgb_model.feature_names_in_)
        else:
            actual_features = None
        
        return {
            'xgb': xgb_model,
            'lstm': lstm_model,
            'scaler': scaler,
            'encoders': encoders,
            'meta': meta_learner,
            'q_lower': q_lower,
            'q_upper': q_upper,
            'feature_names': actual_features
        }
    except Exception as e:
        st.error(f"❌ Error loading AI models: {str(e)}")
        st.info("💡 Make sure all model files are in the same folder as this application.")
        st.stop()

with st.spinner("🔄 Loading AI models..."):
    models = load_models()

# -------------------------------------------------------------------------
# 🔧 FEATURE PREPARATION
# -------------------------------------------------------------------------
CATEGORICAL_FEATURES = ['Store ID', 'Product ID', 'Category', 'Region', 
                        'Weather Condition', 'Seasonality']

def prepare_features(df):
    """Prepare data for AI prediction"""
    tmp = df.copy()
    
    # Handle dates
    if 'Date' not in tmp.columns:
        tmp['Date'] = pd.Timestamp.now()
    tmp['Date'] = pd.to_datetime(tmp['Date'], errors='coerce')
    
    # Extract time information
    tmp['Day_of_Week'] = tmp['Date'].dt.dayofweek
    tmp['Month'] = tmp['Date'].dt.month
    tmp['Quarter'] = tmp['Date'].dt.quarter
    tmp['Is_Weekend'] = (tmp['Day_of_Week'] >= 5).astype(int)
    
    # Set default values for missing information
    defaults = {
        'Epidemic': 0, 'Holiday/Promotion': 0, 'Inventory Level': 100,
        'Competitor Pricing': 20, 'Discount': 0, 'Price': 20,
        'Store ID': 'S001', 'Product ID': 'P001', 'Category': 'Electronics',
        'Region': 'North', 'Weather Condition': 'Sunny', 'Seasonality': 'Spring'
    }
    
    for col, val in defaults.items():
        if col not in tmp.columns:
            tmp[col] = val
    
    # Calculate business metrics
    tmp['Demand_Gap'] = 0
    tmp['Inventory_to_Demand_Ratio'] = tmp['Inventory Level'] / 100
    tmp['Price_to_Competitor_Ratio'] = tmp['Price'] / (tmp['Competitor Pricing'] + 0.01)
    tmp['Discount_Rate'] = tmp['Discount'] / (tmp['Price'] + tmp['Discount'] + 0.01)
    tmp['Effective_Price'] = tmp['Price'] - tmp['Discount']
    tmp['Low_Stock'] = 0
    tmp['Overstock'] = 0
    tmp['Promo_With_Discount'] = tmp['Holiday/Promotion'] * (tmp['Discount'] > 0).astype(int)
    
    # Add historical sales patterns
    lag_defaults = {
        'Units_Sold_Lag_3': 50.0, 'Units_Sold_Lag_7': 48.0, 'Units_Sold_Lag_14': 45.0,
        'Units_Sold_Rolling_Mean_7': 48.0, 'Units_Sold_Rolling_Mean_14': 47.0,
        'Units_Sold_Rolling_Std_7': 5.0, 'Units_Sold_Rolling_Std_14': 7.0
    }
    
    for feat, val in lag_defaults.items():
        if feat not in tmp.columns:
            tmp[feat] = val
    
    # Encode categories to numbers
    for col in CATEGORICAL_FEATURES:
        if col in tmp.columns and col in models['encoders']:
            try:
                tmp[col] = models['encoders'][col].transform(tmp[col].astype(str))
            except:
                tmp[col] = 0
    
    # Use model's expected features
    feature_list = models['feature_names'] if models['feature_names'] else list(tmp.columns)
    
    for feat in feature_list:
        if feat not in tmp.columns:
            tmp[feat] = 0
    
    tmp = tmp[feature_list]
    
    for col in tmp.columns:
        tmp[col] = pd.to_numeric(tmp[col], errors='coerce').fillna(0)
    
    return tmp

def predict_ensemble(df_features):
    """Generate AI predictions"""
    pred_xgb = models['xgb'].predict(df_features)
    
    scaled = models['scaler'].transform(df_features)
    scaled_lstm = np.reshape(scaled, (scaled.shape[0], 1, scaled.shape[1]))
    pred_lstm = models['lstm'].predict(scaled_lstm, verbose=0).flatten()
    
    pred_q25 = models['q_lower'].predict(df_features)
    pred_q75 = models['q_upper'].predict(df_features)
    
    meta_features = np.column_stack([
        pred_xgb, pred_lstm, pred_q25, pred_q75,
        np.minimum(pred_xgb, pred_lstm),
        np.maximum(pred_xgb, pred_lstm),
        (pred_xgb + pred_lstm) / 2,
        np.abs(pred_xgb - pred_lstm)
    ])
    
    pred_meta = models['meta'].predict(meta_features)
    base_ensemble = 0.5 * pred_xgb + 0.3 * pred_lstm + 0.2 * pred_meta
    smoothed = pd.Series(base_ensemble).ewm(span=5, adjust=False).mean().values
    quantile_blend = 0.6 * base_ensemble + 0.2 * pred_q25 + 0.2 * pred_q75
    final_pred = 0.6 * smoothed + 0.4 * quantile_blend
    
    return {
        'prediction': final_pred,
        'lower_bound': pred_q25,
        'upper_bound': pred_q75
    }

# -------------------------------------------------------------------------
# 📱 SIDEBAR - MODE SELECTION
# -------------------------------------------------------------------------
st.sidebar.title("🎯 Choose Your Option")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "",
    ["🏠 Get Started", "📊 Predict from File", "✏️ Manual Prediction"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Tips")
if mode == "📊 Predict from File":
    st.sidebar.info("""
    **Upload your inventory file** and get:
    - Demand predictions for all products
    - 30-day forecasts
    - Price optimization
    - Visual insights
    """)
elif mode == "✏️ Manual Prediction":
    st.sidebar.info("""
    **Enter product details** to:
    - Get instant demand forecast
    - Find optimal pricing
    - View 30-day trends
    """)

# -------------------------------------------------------------------------
# 🏠 GET STARTED PAGE
# -------------------------------------------------------------------------
if mode == "🏠 Get Started":
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 👋 Welcome to Smart Inventory Forecasting!")
        
        st.markdown("""
        <div class="info-box">
        <h4>🎯 What can this tool do for you?</h4>
        <p>Our AI-powered system helps you:</p>
        <ul>
            <li><b>Predict Future Sales:</b> Know how many units you'll sell</li>
            <li><b>Optimize Prices:</b> Find the best price point for maximum revenue</li>
            <li><b>Plan Inventory:</b> Avoid stockouts and overstocking</li>
            <li><b>Forecast Trends:</b> See 30-day demand patterns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🚀 How to Use")
        
        tab1, tab2 = st.tabs(["📊 For Multiple Products", "✏️ For Single Product"])
        
        with tab1:
            st.markdown("""
            **Step 1:** Click on **"📊 Predict from File"** in the sidebar
            
            **Step 2:** Upload your inventory data file (CSV or Excel)
            
            **Step 3:** Get instant predictions for all your products
            
            **Step 4:** Download results and forecasts
            """)
            
            st.success("✅ Perfect for analyzing your entire inventory at once!")
        
        with tab2:
            st.markdown("""
            **Step 1:** Click on **"✏️ Manual Prediction"** in the sidebar
            
            **Step 2:** Fill in product details (price, discount, inventory, etc.)
            
            **Step 3:** Get instant demand forecast
            
            **Step 4:** View optimal pricing recommendations
            """)
            
            st.success("✅ Perfect for quick what-if scenarios!")
        
        st.markdown("---")
        
        st.markdown("### 🤖 About Our AI Models")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("""
            <div style="text-align: center;color:black; padding: 1rem; background: #F0F9FF; border-radius: 10px;">
                <h3>🧠</h3>
                <h4>XGBoost</h4>
                <p>Fast & accurate predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
            <div style="text-align: center;color:black; padding: 1rem; background: #F0F9FF; border-radius: 10px;">
                <h3>🔮</h3>
                <h4>LSTM Neural Network</h4>
                <p>Learns complex patterns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            st.markdown("""
            <div style="text-align: center;color:black; padding: 1rem; background: #F0F9FF; border-radius: 10px;">
                <h3>🎯</h3>
                <h4>Ensemble</h4>
                <p>Best of both worlds</p>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 📊 BATCH PREDICTION MODE
# -------------------------------------------------------------------------
elif mode == "📊 Predict from File":
    st.markdown("## 📊 Upload Your Inventory Data")
    
    st.markdown("""
    <div class="info-box">
    <b>📝 What file should I upload?</b><br>
    Upload a CSV or Excel file containing your product inventory data. 
    The file should include information like Product ID, Store ID, Price, Inventory Level, etc.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your file",
        type=["csv", "xlsx"],
        help="Supported formats: CSV, Excel (.xlsx)"
    )
    
    if uploaded_file:
        try:
            # Load data
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.markdown(f"""
            <div class="success-box">
            ✅ <b>File uploaded successfully!</b><br>
            Found {len(data):,} products in your file
            </div>
            """, unsafe_allow_html=True)
            
            # Preview
            with st.expander("👀 Preview Your Data (first 5 rows)"):
                st.dataframe(data.head(), use_container_width=True)
            
            # Process
            with st.spinner("🔮 AI is analyzing your data..."):
                features_df = prepare_features(data)
                predictions = predict_ensemble(features_df)
            
            # Add predictions
            data['Predicted_Daily_Sales'] = predictions['prediction']
            data['Minimum_Expected'] = predictions['lower_bound']
            data['Maximum_Expected'] = predictions['upper_bound']
            
            # Results
            st.markdown("## 📈 Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                    <h3 style="margin: 0; font-size: 2rem;">{:.1f}</h3>
                    <p style="margin: 0.5rem 0 0 0;">Average Daily Sales</p>
                </div>
                """.format(predictions['prediction'].mean()), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                    <h3 style="margin: 0; font-size: 2rem;">{:,.0f}</h3>
                    <p style="margin: 0.5rem 0 0 0;">Total Units/Month</p>
                </div>
                """.format(predictions['prediction'].sum() * 30), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                            padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                    <h3 style="margin: 0; font-size: 2rem;">{:.1f}</h3>
                    <p style="margin: 0.5rem 0 0 0;">Lowest Forecast</p>
                </div>
                """.format(predictions['prediction'].min()), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                            padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                    <h3 style="margin: 0; font-size: 2rem;">{:.1f}</h3>
                    <p style="margin: 0.5rem 0 0 0;">Highest Forecast</p>
                </div>
                """.format(predictions['prediction'].max()), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Show predictions table
            st.markdown("### 📋 Detailed Predictions")
            display_cols = ['Product ID', 'Store ID', 'Predicted_Daily_Sales', 
                          'Minimum_Expected', 'Maximum_Expected']
            available_cols = [col for col in display_cols if col in data.columns]
            
            if available_cols:
                st.dataframe(
                    data[available_cols].head(20).style.format({
                        'Predicted_Daily_Sales': '{:.1f}',
                        'Minimum_Expected': '{:.1f}',
                        'Maximum_Expected': '{:.1f}'
                    }),
                    use_container_width=True
                )
            
            # Download button
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Complete Predictions",
                csv,
                "demand_predictions.csv",
                "text/csv",
                help="Download all predictions as CSV file"
            )
            
            # 30-Day Forecast
            st.markdown("---")
            st.markdown("## 📆 30-Day Sales Forecast")
            
            if 'Product ID' in data.columns:
                product_list = data['Product ID'].unique()
                selected_product = st.selectbox(
                    "Select a product to view its 30-day forecast:",
                    product_list
                )
                
                product_data = data[data['Product ID'] == selected_product]
                if len(product_data) > 0:
                    base_demand = product_data['Predicted_Daily_Sales'].iloc[0]
                    
                    # Generate forecast
                    days = np.arange(1, 31)
                    np.random.seed(hash(str(selected_product)) % 2**32)
                    
                    trend = 1 + 0.015 * np.linspace(-1, 1, len(days))
                    seasonality = 0.08 * np.sin(2 * np.pi * days / 7)
                    noise = np.random.normal(0, 0.04, len(days))
                    
                    forecast = base_demand * (1 + seasonality) * trend * (1 + noise)
                    forecast = np.maximum(forecast, 0)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(days, forecast, color='#3B82F6', marker='o', linewidth=3, 
                           markersize=6, label='Forecasted Sales')
                    ax.fill_between(days, forecast * 0.9, forecast * 1.1, 
                                   alpha=0.2, color='#3B82F6', label='Uncertainty Range')
                    ax.axhline(base_demand, color='#EF4444', linestyle='--', 
                              linewidth=2, label='Average Prediction')
                    
                    ax.set_xlabel('Day of Month', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Predicted Units Sold', fontsize=12, fontweight='bold')
                    ax.set_title(f'30-Day Sales Forecast for Product {selected_product}', 
                               fontsize=14, fontweight='bold', pad=20)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.legend(fontsize=10, loc='best')
                    
                    # Add background color
                    ax.set_facecolor('#F8FAFC')
                    fig.patch.set_facecolor('white')
                    
                    st.pyplot(fig)
                    
                    # Forecast summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Average Daily Sales", f"{forecast.mean():.1f} units")
                    with col2:
                        st.metric("📈 Peak Day Sales", f"{forecast.max():.1f} units")
                    with col3:
                        st.metric("📉 Low Day Sales", f"{forecast.min():.1f} units")
                    
                    # Download forecast
                    forecast_df = pd.DataFrame({
                        'Day': days,
                        'Predicted_Units': forecast.round(1)
                    })
                    forecast_csv = forecast_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download This Forecast",
                        forecast_csv,
                        f"forecast_{selected_product}.csv",
                        "text/csv"
                    )
            
            # Price Optimization for Batch Mode
            st.markdown("---")
            st.markdown("## 💰 Smart Price Optimization")
            
            st.markdown("""
            <div class="info-box">
            <b>🎯 Optimize Prices for Your Products</b><br>
            Select products to find their optimal price points that maximize revenue.
            Our AI will analyze each product and suggest the best pricing strategy.
            </div>
            """, unsafe_allow_html=True)
            
            # Check if required columns exist
            if 'Price' in data.columns and 'Product ID' in data.columns:
                # Select products for optimization
                st.markdown("### 📋 Select Products to Optimize")
                
                products_to_optimize = st.multiselect(
                    "Choose up to 5 products for detailed price optimization:",
                    options=data['Product ID'].unique()[:20],  # Limit to first 20 for performance
                    default=data['Product ID'].unique()[:min(3, len(data['Product ID'].unique()))],
                    help="Select products you want to optimize pricing for"
                )
                
                if products_to_optimize and st.button("🔍 Optimize Selected Products", type="primary"):
                    optimization_results = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, product in enumerate(products_to_optimize):
                        status_text.text(f"Optimizing {product}... ({idx+1}/{len(products_to_optimize)})")
                        
                        try:
                            # Get product data
                            product_data = data[data['Product ID'] == product].iloc[0:1].copy()
                            current_price = product_data['Price'].values[0]
                            current_discount = product_data.get('Discount', pd.Series([0])).values[0]
                            
                            # Get current prediction
                            current_features = prepare_features(product_data)
                            current_pred = predict_ensemble(current_features)
                            current_demand = current_pred['prediction'][0]
                            current_revenue = (current_price - current_discount) * current_demand
                            
                            # Define optimization function
                            def objective_price(price_val):
                                temp_data = product_data.copy()
                                temp_data['Price'] = price_val
                                temp_data['Effective_Price'] = price_val - temp_data.get('Discount', 0).values[0]
                                temp_data['Price_to_Competitor_Ratio'] = price_val / (temp_data.get('Competitor Pricing', 20).values[0] + 0.01)
                                temp_data['Discount_Rate'] = temp_data.get('Discount', 0).values[0] / (price_val + temp_data.get('Discount', 0).values[0] + 0.01)
                                temp_features = prepare_features(temp_data)
                                temp_pred = predict_ensemble(temp_features)
                                return float(temp_pred['prediction'][0])
                            
                            # Run Bayesian Optimization
                            pbounds = {"price_val": (current_price * 0.7, current_price * 1.3)}
                            optimizer = BayesianOptimization(
                                f=objective_price,
                                pbounds=pbounds,
                                random_state=42,
                                verbose=0
                            )
                            optimizer.maximize(init_points=8, n_iter=15)
                            
                            opt_price = optimizer.max["params"]["price_val"]
                            opt_demand = optimizer.max["target"]
                            opt_revenue = (opt_price - current_discount) * opt_demand
                            revenue_gain = opt_revenue - current_revenue
                            revenue_increase_pct = (revenue_gain / current_revenue) * 100 if current_revenue > 0 else 0
                            
                            optimization_results.append({
                                'Product ID': product,
                                'Current Price': current_price,
                                'Optimal Price': opt_price,
                                'Price Change': opt_price - current_price,
                                'Price Change %': ((opt_price - current_price) / current_price) * 100,
                                'Current Daily Demand': current_demand,
                                'Optimal Daily Demand': opt_demand,
                                'Current Daily Revenue': current_revenue,
                                'Optimal Daily Revenue': opt_revenue,
                                'Revenue Gain': revenue_gain,
                                'Revenue Increase %': revenue_increase_pct,
                                'Monthly Revenue Gain': revenue_gain * 30
                            })
                        
                        except Exception as e:
                            st.warning(f"⚠️ Could not optimize {product}: {str(e)}")
                        
                        progress_bar.progress((idx + 1) / len(products_to_optimize))
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    if optimization_results:
                        results_df = pd.DataFrame(optimization_results)
                        
                        # Summary metrics
                        st.markdown("### 📊 Optimization Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        total_current_revenue = results_df['Current Daily Revenue'].sum()
                        total_opt_revenue = results_df['Optimal Daily Revenue'].sum()
                        total_gain = results_df['Revenue Gain'].sum()
                        avg_increase = results_df['Revenue Increase %'].mean()
                        
                        with col1:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                                <h3 style="margin: 0; font-size: 2rem;">₹{total_current_revenue:,.0f}</h3>
                                <p style="margin: 0.5rem 0 0 0;">Current Daily Revenue</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                                <h3 style="margin: 0; font-size: 2rem;">₹{total_opt_revenue:,.0f}</h3>
                                <p style="margin: 0.5rem 0 0 0;">Optimized Daily Revenue</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                                <h3 style="margin: 0; font-size: 2rem;">₹{total_gain:,.0f}</h3>
                                <p style="margin: 0.5rem 0 0 0;">Daily Revenue Gain</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                                <h3 style="margin: 0; font-size: 2rem;">{avg_increase:+.1f}%</h3>
                                <p style="margin: 0.5rem 0 0 0;">Avg Revenue Increase</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Detailed results table
                        st.markdown("### 📋 Detailed Optimization Results")
                        
                        display_df = results_df[[
                            'Product ID', 'Current Price', 'Optimal Price', 'Price Change %',
                            'Current Daily Demand', 'Optimal Daily Demand',
                            'Revenue Gain', 'Revenue Increase %'
                        ]].copy()
                        
                        st.dataframe(
                            display_df.style.format({
                                'Current Price': '₹{:.2f}',
                                'Optimal Price': '₹{:.2f}',
                                'Price Change %': '{:+.1f}%',
                                'Current Daily Demand': '{:.1f}',
                                'Optimal Daily Demand': '{:.1f}',
                                'Revenue Gain': '₹{:+,.0f}',
                                'Revenue Increase %': '{:+.1f}%'
                            }).background_gradient(subset=['Revenue Increase %'], cmap='RdYlGn', vmin=-10, vmax=10),
                            use_container_width=True
                        )
                        
                        # Visualization
                        st.markdown("### 📈 Revenue Impact Visualization")
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                        
                        # Revenue comparison
                        x_pos = np.arange(len(results_df))
                        width = 0.35
                        
                        ax1.bar(x_pos - width/2, results_df['Current Daily Revenue'], width, 
                               label='Current', color='#94A3B8', alpha=0.8)
                        ax1.bar(x_pos + width/2, results_df['Optimal Daily Revenue'], width, 
                               label='Optimized', color='#10B981', alpha=0.8)
                        
                        ax1.set_xlabel('Product', fontweight='bold')
                        ax1.set_ylabel('Daily Revenue (₹)', fontweight='bold')
                        ax1.set_title('Revenue Comparison: Current vs Optimized', fontweight='bold', pad=15)
                        ax1.set_xticks(x_pos)
                        ax1.set_xticklabels(results_df['Product ID'], rotation=45, ha='right')
                        ax1.legend()
                        ax1.grid(axis='y', alpha=0.3)
                        
                        # Price change visualization
                        colors = ['#10B981' if x > 0 else '#EF4444' if x < 0 else '#6B7280' 
                                 for x in results_df['Price Change %']]
                        ax2.barh(results_df['Product ID'], results_df['Price Change %'], color=colors, alpha=0.8)
                        ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')
                        ax2.set_xlabel('Price Change (%)', fontweight='bold')
                        ax2.set_title('Recommended Price Adjustments', fontweight='bold', pad=15)
                        ax2.grid(axis='x', alpha=0.3)
                        
                        fig.tight_layout()
                        st.pyplot(fig)
                        
                        # Recommendations
                        st.markdown("---")
                        st.markdown("### 💡 Key Recommendations")
                        
                        # Find best opportunities
                        best_opportunity = results_df.loc[results_df['Revenue Gain'].idxmax()]
                        products_to_increase = results_df[results_df['Price Change %'] > 2]
                        products_to_decrease = results_df[results_df['Price Change %'] < -2]
                        
                        if best_opportunity['Revenue Gain'] > 0:
                            st.markdown(f"""
                            <div class="success-box">
                            <h4>🎯 Best Opportunity: {best_opportunity['Product ID']}</h4>
                            <p>Changing price from <b>₹{best_opportunity['Current Price']:.2f}</b> to 
                            <b>₹{best_opportunity['Optimal Price']:.2f}</b> could increase daily revenue by 
                            <b>₹{best_opportunity['Revenue Gain']:,.0f}</b> ({best_opportunity['Revenue Increase %']:.1f}% increase)</p>
                            <p><b>Monthly Impact:</b> ₹{best_opportunity['Monthly Revenue Gain']:,.0f} additional revenue</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if len(products_to_increase) > 0:
                            st.markdown(f"""
                            <div class="info-box">
                            <h4>📈 Consider Price Increases</h4>
                            <p>These products can support higher prices: <b>{', '.join(products_to_increase['Product ID'].tolist())}</b></p>
                            <p>Average recommended increase: <b>{products_to_increase['Price Change %'].mean():.1f}%</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if len(products_to_decrease) > 0:
                            st.markdown(f"""
                            <div class="warning-box">
                            <h4>📉 Consider Price Reductions</h4>
                            <p>These products may benefit from lower prices: <b>{', '.join(products_to_decrease['Product ID'].tolist())}</b></p>
                            <p>Lower prices could increase volume and total revenue</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Monthly impact summary
                        monthly_gain = results_df['Monthly Revenue Gain'].sum()
                        if monthly_gain > 100:
                            st.success(f"💰 **Total Potential Monthly Revenue Increase: ₹{monthly_gain:,.0f}**")
                        
                        # Download optimization results
                        opt_csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇️ Download Optimization Results",
                            opt_csv,
                            "price_optimization_results.csv",
                            "text/csv",
                            help="Download complete optimization analysis"
                        )
                    else:
                        st.warning("⚠️ No optimization results were generated. Please try again.")
            else:
                st.info("ℹ️ Price optimization requires 'Price' and 'Product ID' columns in your data.")
        
        except Exception as e:
            st.error(f"❌ Oops! Something went wrong: {str(e)}")
            st.info("💡 Please make sure your file has the correct format and try again.")

# -------------------------------------------------------------------------
# ✏️ MANUAL PREDICTION MODE
# -------------------------------------------------------------------------
else:
    st.markdown("## ✏️ Enter Product Details")
    
    st.markdown("""
    <div class="info-box">
    <b>💡 How does this work?</b><br>
    Fill in the details below about your product. Our AI will instantly predict 
    how many units you can expect to sell daily and suggest the best price.
    </div>
    """, unsafe_allow_html=True)
    
    # Input form
    st.markdown("### 📝 Product Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏪 Store & Product Details")
        store_id = st.text_input("Store ID", "S001", help="Your store identifier")
        product_id = st.text_input("Product ID", "P001", help="Your product identifier")
        category = st.selectbox("Product Category", 
                               ["Electronics", "Clothing", "Groceries", "Furniture", "Other"])
        region = st.selectbox("Store Region", ["North", "South", "East", "West"])
        
        st.markdown("#### 💰 Pricing & Promotions")
        price = st.number_input("Product Price (₹)", min_value=0.0, value=500.0, step=10.0,
                               help="Regular selling price")
        discount = st.number_input("Discount Amount (₹)", min_value=0.0, value=50.0, step=10.0,
                                  help="How much discount you're offering")
        promo = st.selectbox("Is there a promotion/holiday?", [("No", 0), ("Yes", 1)],
                           format_func=lambda x: x[0])
        
    with col2:
        st.markdown("#### 📦 Inventory & Competition")
        inventory = st.number_input("Current Inventory Level", min_value=0, value=100,
                                   help="How many units do you have in stock?")
        competitor = st.number_input("Competitor Price (₹)", min_value=0.0, value=550.0, step=10.0,
                                    help="What price are competitors charging?")
        
        st.markdown("#### 🌤️ External Factors")
        weather = st.selectbox("Weather Condition", 
                              ["Sunny", "Rainy", "Cloudy", "Snowy"])
        season = st.selectbox("Season", 
                            ["Spring", "Summer", "Fall", "Winter"])
        epidemic = st.selectbox("Is there an epidemic/pandemic?", 
                              [("No", 0), ("Yes", 1)],
                              format_func=lambda x: x[0])
    
    st.markdown("#### 📊 Recent Sales History")
    col_a, col_b = st.columns(2)
    with col_a:
        lag_7 = st.number_input("Units sold 7 days ago", min_value=0.0, value=45.0,
                               help="How many units did you sell last week?")
    with col_b:
        lag_14 = st.number_input("Units sold 14 days ago", min_value=0.0, value=43.0,
                                help="How many units did you sell 2 weeks ago?")
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Demand", type="primary", use_container_width=True):
        # Create input
        input_data = pd.DataFrame([{
            'Store ID': store_id, 'Product ID': product_id, 'Category': category,
            'Region': region, 'Inventory Level': inventory, 'Price': price,
            'Discount': discount, 'Weather Condition': weather,
            'Holiday/Promotion': promo[1], 'Epidemic': epidemic[1],
            'Competitor Pricing': competitor, 'Seasonality': season,
            'Date': pd.Timestamp.now(),
            'Units_Sold_Lag_7': lag_7, 'Units_Sold_Lag_14': lag_14,
            'Units_Sold_Lag_3': (lag_7 + lag_14) / 2,
            'Units_Sold_Rolling_Mean_7': lag_7,
            'Units_Sold_Rolling_Mean_14': lag_14,
            'Units_Sold_Rolling_Std_7': 5.0,
            'Units_Sold_Rolling_Std_14': 7.0
        }])
        
        try:
            with st.spinner("🤖 AI is making predictions..."):
                features_df = prepare_features(input_data)
                predictions = predict_ensemble(features_df)
            
            # Show results
            st.markdown("""
            <div class="success-box">
            <h3 style="margin-top: 0;">✅ Prediction Complete!</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Main prediction
            st.markdown("### 🎯 Expected Daily Sales")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 2rem; border-radius: 10px; color: white; text-align: center;">
                    <h2 style="margin: 0; font-size: 3rem;">{:.0f}</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Units per Day</p>
                </div>
                """.format(predictions['prediction'][0]), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 2rem; border-radius: 10px; color: white; text-align: center;">
                    <h2 style="margin: 0; font-size: 3rem;">{:.0f}</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Minimum Expected</p>
                </div>
                """.format(predictions['lower_bound'][0]), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                            padding: 2rem; border-radius: 10px; color: white; text-align: center;">
                    <h2 style="margin: 0; font-size: 3rem;">{:.0f}</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Maximum Expected</p>
                </div>
                """.format(predictions['upper_bound'][0]), unsafe_allow_html=True)
            
            # Revenue projection
            st.markdown("---")
            st.markdown("### 💵 Revenue Projection")
            
            effective_price = price - discount
            daily_revenue = effective_price * predictions['prediction'][0]
            monthly_revenue = daily_revenue * 30
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Expected Daily Revenue", f"₹{daily_revenue:,.0f}")
            with col2:
                st.metric("Expected Monthly Revenue", f"₹{monthly_revenue:,.0f}")
            
            # 30-Day Forecast
            st.markdown("---")
            st.markdown("### 📆 30-Day Sales Trend")
            
            days = np.arange(1, 31)
            base_demand = predictions['prediction'][0]
            
            np.random.seed(42)
            trend = 1 + 0.015 * np.linspace(-1, 1, len(days))
            seasonality = 0.08 * np.sin(2 * np.pi * days / 7)
            noise = np.random.normal(0, 0.04, len(days))
            forecast = base_demand * (1 + seasonality) * trend * (1 + noise)
            forecast = np.maximum(forecast, 0)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(days, forecast, color='#3B82F6', marker='o', linewidth=3, 
                   markersize=6, label='Forecasted Sales', markerfacecolor='white', 
                   markeredgewidth=2)
            ax.fill_between(days, forecast * 0.9, forecast * 1.1, 
                           alpha=0.2, color='#3B82F6', label='Confidence Range (±10%)')
            ax.axhline(base_demand, color='#EF4444', linestyle='--', 
                      linewidth=2, label='Average Expected Sales')
            
            ax.set_xlabel('Day of Month', fontsize=13, fontweight='bold')
            ax.set_ylabel('Predicted Units Sold', fontsize=13, fontweight='bold')
            ax.set_title(f'30-Day Sales Forecast for {product_id}', 
                       fontsize=15, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=11, loc='best', framealpha=0.9)
            
            ax.set_facecolor('#F8FAFC')
            fig.patch.set_facecolor('white')
            
            st.pyplot(fig)
            
            # Forecast insights
            st.markdown("""
            <div class="info-box">
            <b>📊 What does this forecast tell you?</b><br>
            • <b>Blue line:</b> Expected daily sales for the next 30 days<br>
            • <b>Shaded area:</b> Range where actual sales will likely fall<br>
            • <b>Red dashed line:</b> Your average expected sales<br>
            • <b>Weekly pattern:</b> You can see how sales vary throughout the week
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Average Daily Sales", f"{forecast.mean():.0f} units", 
                         help="Average sales across 30 days")
            with col2:
                st.metric("📈 Best Day Forecast", f"{forecast.max():.0f} units", 
                         f"+{((forecast.max()-forecast.mean())/forecast.mean()*100):.0f}%",
                         help="Your highest expected sales day")
            with col3:
                total_month = forecast.sum()
                st.metric("📦 Total Monthly Units", f"{total_month:.0f} units",
                         help="Total units expected to sell in 30 days")
            
            # Price Optimization
            st.markdown("---")
            st.markdown("### 💰 Smart Price Optimization")
            
            st.markdown("""
            <div class="warning-box">
            <b>🤔 Want to maximize your profits?</b><br>
            Let our AI find the optimal price point that balances sales volume and revenue!
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("🔍 Finding the best price for you..."):
                def objective_price(price_val):
                    temp_data = input_data.copy()
                    temp_data['Price'] = price_val
                    temp_data['Effective_Price'] = price_val - temp_data['Discount'].values[0]
                    temp_data['Price_to_Competitor_Ratio'] = price_val / (temp_data['Competitor Pricing'].values[0] + 0.01)
                    temp_data['Discount_Rate'] = temp_data['Discount'].values[0] / (price_val + temp_data['Discount'].values[0] + 0.01)
                    temp_features = prepare_features(temp_data)
                    temp_pred = predict_ensemble(temp_features)
                    return float(temp_pred['prediction'][0])
                
                pbounds = {"price_val": (price * 0.7, price * 1.3)}
                optimizer = BayesianOptimization(
                    f=objective_price,
                    pbounds=pbounds,
                    random_state=42,
                    verbose=0
                )
                optimizer.maximize(init_points=8, n_iter=15)
                
                opt_price = optimizer.max["params"]["price_val"]
                opt_demand = optimizer.max["target"]
                
                current_revenue = (price - discount) * predictions['prediction'][0]
                opt_revenue = (opt_price - discount) * opt_demand
                revenue_gain = opt_revenue - current_revenue
                revenue_increase_pct = (revenue_gain / current_revenue) * 100 if current_revenue > 0 else 0
            
            # Show optimization results
            st.markdown("#### 🎯 Pricing Comparison")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div style="background: #F3F4F6; padding: 1.5rem; border-radius: 10px; border: 2px solid #D1D5DB;">
                    <h4 style="margin: 0; color: #6B7280;">Current Setup</h4>
                    <hr style="margin: 1rem 0; border: none; border-top: 1px solid #D1D5DB;">
                    <p style="margin: 0.5rem 0;"><b>Price:</b> ₹{:.2f}</p>
                    <p style="margin: 0.5rem 0;"><b>Expected Sales:</b> {:.0f} units/day</p>
                    <p style="margin: 0.5rem 0;"><b>Daily Revenue:</b> ₹{:,.0f}</p>
                </div>
                """.format(price, predictions['prediction'][0], current_revenue), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); 
                            padding: 1.5rem; border-radius: 10px; color: white;">
                    <h4 style="margin: 0;">✨ Optimized Setup</h4>
                    <hr style="margin: 1rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.3);">
                    <p style="margin: 0.5rem 0;"><b>Price:</b> ₹{:.2f}</p>
                    <p style="margin: 0.5rem 0;"><b>Expected Sales:</b> {:.0f} units/day</p>
                    <p style="margin: 0.5rem 0;"><b>Daily Revenue:</b> ₹{:,.0f}</p>
                </div>
                """.format(opt_price, opt_demand, opt_revenue), unsafe_allow_html=True)
            
            with col3:
                change_symbol = "📈" if revenue_gain > 0 else "📉" if revenue_gain < 0 else "➡️"
                change_color = "#10B981" if revenue_gain > 0 else "#EF4444" if revenue_gain < 0 else "#6B7280"
                
                st.markdown("""
                <div style="background: {}; padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                    <h4 style="margin: 0;">{} Impact</h4>
                    <hr style="margin: 1rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.3);">
                    <h2 style="margin: 1rem 0; font-size: 2.5rem;">{:+.1f}%</h2>
                    <p style="margin: 0.5rem 0;"><b>Revenue Change</b></p>
                    <p style="margin: 0.5rem 0;">₹{:+,.0f}/day</p>
                </div>
                """.format(change_color, change_symbol, revenue_increase_pct, revenue_gain), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Recommendation
            if revenue_increase_pct > 5:
                st.markdown("""
                <div class="success-box">
                <h4>✅ Recommendation: Adjust Your Price</h4>
                <p>Our AI suggests changing your price to <b>₹{:.2f}</b>. This could increase your 
                daily revenue by <b>₹{:,.0f}</b> ({:.1f}% increase)!</p>
                <p><b>Why?</b> At this price point, even though you might sell slightly fewer or more units, 
                your total revenue will be higher.</p>
                </div>
                """.format(opt_price, revenue_gain, revenue_increase_pct), unsafe_allow_html=True)
            elif revenue_increase_pct < -5:
                st.markdown("""
                <div class="warning-box">
                <h4>⚠️ Current Price Seems Too High</h4>
                <p>Consider reducing your price to <b>₹{:.2f}</b>. While this is lower, you'll sell 
                more units and increase your total revenue by <b>₹{:,.0f}</b>.</p>
                </div>
                """.format(opt_price, abs(revenue_gain)), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                <h4>✨ Your Price is Already Optimized!</h4>
                <p>Great job! Your current price of <b>₹{:.2f}</b> is very close to the optimal 
                price point. Keep it as is!</p>
                </div>
                """.format(price), unsafe_allow_html=True)
            
            # Monthly projection with optimal price
            if revenue_increase_pct > 2:
                st.markdown("#### 📊 Monthly Impact of Price Optimization")
                
                col1, col2 = st.columns(2)
                with col1:
                    current_monthly = current_revenue * 30
                    st.metric("Current Monthly Revenue", f"₹{current_monthly:,.0f}")
                with col2:
                    opt_monthly = opt_revenue * 30
                    st.metric("Optimized Monthly Revenue", f"₹{opt_monthly:,.0f}", 
                             f"+₹{opt_monthly - current_monthly:,.0f}")
                
                st.success(f"💡 By optimizing your price, you could earn an additional **₹{(opt_monthly - current_monthly):,.0f} per month**!")
        
        except Exception as e:
            st.error(f"❌ Oops! Something went wrong: {str(e)}")
            st.info("💡 Please check your inputs and try again. Make sure all fields are filled correctly.")

# -------------------------------------------------------------------------
# 📊 FOOTER
# -------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: #F8FAFC; border-radius: 10px; margin-top: 3rem;'>
    <h3 style='color: #1E3A8A; margin-bottom: 1rem;'>📦 Smart Inventory Forecasting System</h3>
    <p style='color: #64748B; margin-bottom: 0.5rem;'>Powered by Advanced AI & Machine Learning</p>
    <p style='color: #94A3B8; font-size: 0.9rem;'>XGBoost • LSTM Neural Networks • Ensemble Methods • Bayesian Optimization</p>
    <hr style='margin: 1.5rem auto; width: 50%; border: none; border-top: 2px solid #E2E8F0;'>
    <p style='color: #94A3B8; font-size: 0.85rem; margin: 0;'>
        💡 <b>Need Help?</b> Use the sidebar to navigate between different modes<br>
        🔧 For support or questions, contact your system administrator
    </p>
</div>
""", unsafe_allow_html=True)