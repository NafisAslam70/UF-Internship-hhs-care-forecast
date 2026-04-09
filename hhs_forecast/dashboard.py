# dashboard.py - with discharge debug
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent

# ============================================
# 1. Load original data and preprocess
# ============================================
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv(BASE_DIR / 'prjct1_data.csv')
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.groupby('Date').last().reset_index()
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    numeric_cols = [col for col in df.columns if 'CBP' in col or 'HHS' in col or 'discharged' in col]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    expected_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(expected_dates)
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(0)
    
    target_col = 'Children in HHS Care'
    discharge_col = 'Children discharged from HHS Care'
    
    df['Lag_1'] = df[target_col].shift(1)
    df['Lag_7'] = df[target_col].shift(7)
    df['Lag_14'] = df[target_col].shift(14)
    df['Rolling_7'] = df[target_col].rolling(7).mean()
    df['Rolling_14'] = df[target_col].rolling(14).mean()
    df['Net Pressure'] = df['Children transferred out of CBP custody'] - df[discharge_col]
    df['DayOfWeek'] = df.index.dayofweek
    
    df_clean = df.dropna()
    return df_clean, target_col, discharge_col

df, target_col, discharge_col = load_and_preprocess()

features = ['Lag_1', 'Lag_7', 'Lag_14', 'Rolling_7', 'Rolling_14', 'Net Pressure', 'DayOfWeek']

split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
rf_full.fit(df[features], df[target_col])

def persistence_forecast(last_value, horizon):
    return np.full(horizon, last_value)

# ============================================
# 2. Streamlit UI
# ============================================
st.title("📊 HHS Care Load Forecasting Dashboard")
st.markdown("Predictive forecasting of children in HHS care and discharge demand")

# Debug expander (shows discharge data)
with st.expander("🔍 Debug: Check discharge data"):
    st.write(f"Discharge column name: `{discharge_col}`")
    st.write("Last 10 discharge values:")
    st.write(df[discharge_col].tail(10))
    st.write(f"Last discharge value: {df[discharge_col].iloc[-1]}")

# Sidebar
st.sidebar.header("Model Settings")
horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 7)
model_choice = st.sidebar.selectbox("Select Model", ["Persistence (Naïve)", "Random Forest", "SARIMA"])

# Main forecast
st.subheader("📈 Future Care Load Forecast")
last_value = df[target_col].iloc[-1]
last_date = df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')

if model_choice == "Persistence (Naïve)":
    forecast = persistence_forecast(last_value, horizon)
    st.info("Using naïve persistence: tomorrow = today. Best for near-random-walk series.")
elif model_choice == "Random Forest":
    last_features = df[features].iloc[-1:].values
    forecast = []
    current_features = last_features[0].copy()
    for _ in range(horizon):
        pred = rf_full.predict([current_features])[0]
        forecast.append(pred)
        lag1_idx = features.index('Lag_1')
        current_features[lag1_idx] = pred
    st.warning("Multi-step RF forecast is approximate.")
else:
    model_sarima = SARIMAX(df[target_col], order=(1,0,1), seasonal_order=(1,0,1,7),
                           enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = model_sarima.fit(disp=False)
    forecast = sarima_fit.forecast(steps=horizon)
    st.info("SARIMA with weekly seasonality.")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index[-90:], df[target_col].iloc[-90:], label='Historical', color='blue')
ax.plot(forecast_dates, forecast, label=f'{model_choice} Forecast', color='red', linestyle='--')
ax.fill_between(forecast_dates, np.array(forecast)*0.95, np.array(forecast)*1.05, alpha=0.2, color='red')
ax.set_title(f'{model_choice} Forecast - Next {horizon} Days')
ax.set_xlabel('Date')
ax.set_ylabel('Children in HHS Care')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Discharge demand panel
# Discharge demand panel (fixed for visibility)
st.subheader("🚪 Discharge Demand Forecast")
discharge_last = df[discharge_col].iloc[-1]
discharge_forecast = persistence_forecast(discharge_last, horizon)

# Debug: show values
with st.expander("🔍 Discharge forecast values"):
    st.write(f"Last actual discharge: {discharge_last}")
    st.write(f"Forecast horizon (days): {horizon}")
    st.write(f"Forecasted daily discharges: {discharge_forecast}")

# Plot with explicit markers and line
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(forecast_dates, discharge_forecast, marker='o', linestyle='-', color='green', linewidth=2, markersize=6, label='Forecasted Discharges')
ax2.set_title(f'Discharge Demand (Persistence Forecast) – Next {horizon} Days')
ax2.set_xlabel('Date')
ax2.set_ylabel('Number of Discharges')
ax2.grid(True, alpha=0.3)
ax2.legend()
st.pyplot(fig2)

# Model comparison
st.subheader("📊 Model Comparison on Historical Test Set")
y_test = test[target_col]
y_pred_persist = [train[target_col].iloc[-1]] + list(y_test[:-1])
mae_persist = np.mean(np.abs(y_test - y_pred_persist))
X_test = test[features]
y_pred_rf = rf_full.predict(X_test)
mae_rf = np.mean(np.abs(y_test - y_pred_rf))
model_sarima_test = SARIMAX(train[target_col], order=(1,0,1), seasonal_order=(1,0,1,7))
sarima_fit_test = model_sarima_test.fit(disp=False)
y_pred_sarima = sarima_fit_test.forecast(steps=len(y_test))
mae_sarima = np.mean(np.abs(y_test - y_pred_sarima))

comparison = pd.DataFrame({
    'Model': ['Persistence', 'Random Forest', 'SARIMA'],
    'MAE (test)': [mae_persist, mae_rf, mae_sarima]
})
st.table(comparison)
st.markdown("**Note:** Lower MAE is better. Persistence often outperforms complex models on stable series.")

# Early warning
st.subheader("⚠️ Capacity Stress Early Warning")
net_pressure = df['Net Pressure'].iloc[-7:].mean()
if net_pressure > 50:
    st.error(f"High net inflow pressure ({net_pressure:.0f} avg last 7 days). Consider scaling up shelters.")
elif net_pressure > 20:
    st.warning(f"Moderate net inflow pressure ({net_pressure:.0f}). Monitor closely.")
else:
    st.success(f"Stable net pressure ({net_pressure:.0f}).")

st.markdown("---")
st.caption("Dashboard for HHS UAC Program - Predictive Forecasting")
