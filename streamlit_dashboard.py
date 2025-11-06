import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout='wide', page_title='Traffic Prediction Dashboard')

st.title('Traffic Prediction Dashboard')

# --- Utilities ---
@st.cache_data
def load_data(path):
    df = pd.read_csv("/Users/serengeti/Downloads/traffic.csv")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    return df

def Normalize(df, col):
    avg = df[col].mean()
    std = df[col].std()
    df_norm = (df[col] - avg) / std
    return df_norm.to_frame(), avg, std

def Difference(df, col, interval):
    diff = []
    for i in range(interval, len(df)):
        diff.append(df[col].iloc[i] - df[col].iloc[i - interval])
    return diff

def reconstruct_predictions(raw_df, mean, std, interval, pred_arr):
    if isinstance(raw_df, pd.DataFrame):
        raw_vals = raw_df['Vehicles'].astype(float).values
    else:
        raw_vals = raw_df.astype(float).values
    norm_full = (raw_vals - mean) / std
    k = interval
    diff_obs = norm_full[k:] - norm_full[:-k]
    m = len(diff_obs)
    p = len(pred_arr)
    start_replace = m - p
    diff_combined = diff_obs.copy()
    diff_combined[start_replace:start_replace + p] = pred_arr
    norm_recon = norm_full.copy()
    for t in range(m):
        j = t + k
        norm_recon[j] = diff_combined[t] + norm_recon[t]
    norm_pred_values = norm_recon[k + start_replace:k + m]
    real_pred_values = (norm_pred_values * std) + mean
    return real_pred_values

def simple_auto_arima(train):
    """Very small manual search to approximate auto_arima."""
    best_order, best_aic = None, np.inf
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    model = ARIMA(train, order=(p, d, q)).fit()
                    if model.aic < best_aic:
                        best_aic, best_order = model.aic, (p, d, q)
                except:
                    continue
    return best_order or (1, 0, 1)

# --- Sidebar controls ---
st.sidebar.header('Controls')
default_path = '/Users/diya/Downloads/traffic.csv'
file_path = st.sidebar.text_input('CSV path', value=default_path)
n_last = st.sidebar.number_input('Show last N points in plots', min_value=10, max_value=1000, value=200)
auto_select = st.sidebar.checkbox('Run simple auto-ARIMA (statsmodels)', value=True)

# --- Load data ---
with st.spinner('Loading data...'):
    try:
        data = load_data(file_path)
    except Exception as e:
        st.error(f'Failed to load CSV: {e}')
        st.stop()

st.markdown('## Raw data')
st.dataframe(data.head())

col1, col2, col3 = st.columns(3)
col1.metric('Rows', len(data))
col2.metric('Start', str(data['DateTime'].min()))
col3.metric('End', str(data['DateTime'].max()))

pivot = data.pivot(index='DateTime', columns='Junction', values='Vehicles')

st.markdown('## Time series overview')
fig = px.line(data, x='DateTime', y='Vehicles', color='Junction', title='Vehicles over time by Junction')
st.plotly_chart(fig, use_container_width=True)

# --- Junction selector ---
junctions = sorted(data['Junction'].unique())
sel_junction = st.sidebar.selectbox('Select junction', junctions)
default_intervals = {1: 24*7, 2: 24, 3: 1, 4: 1}
interval = st.sidebar.number_input('Differencing interval (hours)', min_value=1, value=int(default_intervals.get(sel_junction, 24)))

raw_df = pivot[[sel_junction]].rename(columns={sel_junction: 'Vehicles'}).dropna()

st.markdown(f'### Junction {sel_junction} — raw series')
st.line_chart(raw_df.tail(n_last))

# --- Normalization & differencing ---
df_N, av, std = Normalize(raw_df, 'Vehicles')
diff_vals = Difference(df_N, 'Vehicles', interval)
df_N_proc = df_N[interval:].copy()
df_N_proc.columns = ['Norm']
df_N_proc['Diff'] = diff_vals

st.markdown('### Normalized and Differenced series (sample)')
st.dataframe(df_N_proc.head())

st.markdown('### Differenced (used for modeling)')
fig2 = px.line(df_N_proc['Diff'].reset_index(drop=True).rename('Diff'), y='Diff', title='Differenced series')
st.plotly_chart(fig2, use_container_width=True)

# --- Train/Test split ---
train_size = int(len(df_N_proc) * 0.9)
train = df_N_proc['Diff'].iloc[:train_size]
test = df_N_proc['Diff'].iloc[train_size:]

st.markdown(f'Training points: {len(train)} — Test points: {len(test)}')

order_input = st.sidebar.text_input('ARIMA order (p,d,q) manual override — e.g. 2,0,2', value='')

if order_input.strip():
    order = tuple(int(x.strip()) for x in order_input.split(','))
else:
    if auto_select:
        with st.spinner('Finding best ARIMA order...'):
            order = simple_auto_arima(train)
            st.sidebar.write('Best order (approx.):', order)
    else:
        order = (1, 0, 1)

st.write('Using ARIMA order:', order)

# --- Fit ARIMA ---
with st.spinner('Fitting ARIMA...'):
    try:
        model = ARIMA(df_N_proc['Diff'], order=order)
        results = model.fit()
        st.subheader('ARIMA summary (trimmed)')
        st.text('\n'.join(results.summary().as_text().split('\n')[:10]))
    except Exception as e:
        st.error(f'ARIMA fit failed: {e}')
        results = None

# --- Forecast & reconstruction ---
if results is not None:
    n_forecast = min(100, len(df_N_proc))
    start, end = len(df_N_proc) - n_forecast, len(df_N_proc) - 1
    pred = results.predict(start=start, end=end)
    pred.index = df_N_proc.index[start:end+1]
    pred_arr = np.array(pred).ravel()

    try:
        real_preds = reconstruct_predictions(raw_df, av, std, interval, pred_arr)
        pred_index = raw_df.index[start+interval:start+interval+len(pred_arr)]
        real_series = pd.Series(real_preds, index=pred_index, name='predicted')

        actual_on_pred = raw_df.loc[pred_index, 'Vehicles']
        mae = mean_absolute_error(actual_on_pred, real_series)
        rmse = np.sqrt(mean_squared_error(actual_on_pred, real_series))
        

        st.subheader('Actual vs Predicted (reconstructed real counts)')
        fig3 = go.Figure()
        # Only plot recent data for better performance
        recent_actual = raw_df.tail(n_last)
        fig3.add_trace(go.Scatter(x=recent_actual.index, y=recent_actual['Vehicles'], mode='lines', name='Actual', opacity=0.5))
        recent_pred = real_series[real_series.index >= recent_actual.index[0]]
        fig3.add_trace(go.Scatter(x=recent_pred.index, y=recent_pred.values, mode='lines+markers', name='Predicted', line=dict(color='red')))
        fig3.update_layout(title=f'Junction {sel_junction} — Actual vs Predicted (MAE={mae:.2f}, RMSE={rmse:.2f})', xaxis_title='DateTime', yaxis_title='Vehicles')
        st.plotly_chart(fig3, use_container_width=True)

        st.write('MAE:', mae)
        st.write('RMSE:', rmse)

    except Exception as e:
        st.error(f'Failed to reconstruct predictions: {e}')

st.markdown('---')
st.write('You can change junction, differencing interval, ARIMA order or upload a different CSV using the sidebar.')
