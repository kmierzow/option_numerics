import streamlit as st
import numerical_methods
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import plotly.graph_objs as go
from scipy.interpolate import griddata

def calculate_time_to_expiration(expiration_date):
    exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
    today = datetime.now()
    days_to_exp = (exp_date - today).days
    return max(days_to_exp / 365.0, 1/365)

def calculate_time_to_expiration_vectorized(date_str_series):
    today = datetime.now()
    return (pd.to_datetime(date_str_series) - today).dt.days.clip(lower=1) / 365.0

@st.cache_data
def calculate_IV_surface(ticker_input, r, t_steps, S_steps, max_iterations):
    ticker = yf.Ticker(ticker_input)
    current_price = ticker.history(period="1d")['Close'].iloc[-1]
    div_yield = ticker.info.get("dividendYield") or 0
    S_max = current_price*2 


    expirations = ticker.options
    options_data = []
    for expiration_date in expirations:
        options_chain = ticker.option_chain(expiration_date)

        for opt_type, data in [('call', options_chain.calls), ('put', options_chain.puts)]:
            data = data[data['volume'] > 0]
            data = data[['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']].copy()
            data['expiration'] = expiration_date
            data['type'] = opt_type
            options_data.append(data)

    input_data = pd.concat(options_data, ignore_index=True)
    input_data['expiration'] = calculate_time_to_expiration_vectorized(input_data['expiration'])

    output_data = pd.DataFrame({
        'strike price': input_data['strike'],
        'expiration': input_data['expiration'],
        'implied volatility': numerical_methods.sigma_impl_solver_batch(
            input_data['strike'].to_numpy(dtype=np.float64),
            r,
            div_yield,
            input_data['lastPrice'].to_numpy(dtype=np.float64),
            input_data['expiration'].to_numpy(dtype=np.float64),
            t_steps,
            S_max, 
            S_steps,
            input_data['type'] == 'call',
            max_iterations,
            1e-6
        ),
        'type': input_data['type']
    }).dropna()
    return output_data


st.title("Option analysis metrics")
r = 0.0435

ticker_input = st.text_input("Ticker:", "NVDA")


with st.sidebar:
    t_steps = st.number_input("Input the amount of time steps", value=10000)
    S_steps = st.number_input("Input the amount of underlying asset prices", value=25)
    max_iterations = 100

underlying_price =  yf.Ticker(ticker_input).history(period="1d")['Close'].iloc[-1]

output_data = calculate_IV_surface(ticker_input, r, t_steps, S_steps, max_iterations)
output_data['moneyness'] = output_data['strike price'] / underlying_price
grid_moneyness = np.linspace(output_data['moneyness'].min(), 
                            output_data['moneyness'].max(), 
                            50)
grid_exp = np.linspace(output_data['expiration'].min(), 
                      output_data['expiration'].max(), 
                      50)
X, Y = np.meshgrid(grid_moneyness, grid_exp)

Z = griddata(
    (output_data['moneyness'], output_data['expiration']),
    output_data['implied volatility'],
    (X, Y),
    method='nearest'
)

surface = go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.7)

points = go.Scatter3d(
    x=output_data['moneyness'],
    y=output_data['expiration'],
    z=output_data['implied volatility'],
    mode='markers',
    marker=dict(size=3, color='red'),
    name="Data Points"
)

fig = go.Figure(data=[surface, points])

fig.update_layout(
    title='Implied Volatility Surface',
    scene=dict(
        xaxis_title='Moneyness (Strike/Current Price)',
        yaxis_title='Time to Expiration (Years)',
        zaxis_title='Implied Volatility'
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)

st.plotly_chart(fig, use_container_width=True)


avalible_dates = output_data['expiration'].unique().tolist()
selected_date = st.selectbox("Select an expiry (in years) for the crossection (volatility smile):", avalible_dates)
date_filtered = output_data.loc[output_data['expiration'] == selected_date]
fig2 = st.scatter_chart(date_filtered.set_index("moneyness")["implied volatility"])

