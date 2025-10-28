import setup
import streamlit as st
import numerical_methods
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Heatmap of option prices")

with st.sidebar:
    st.write("Individual price settings:")
    is_call = st.segmented_control("Option type:", ["Call", "Put"], default="Call") == "Call"
    underlying_price = st.number_input("Underlying price:", value= 100)
    strike_price = st.number_input("Strike price:", value= 100)
    risk_free_rate = st.number_input("Risk free rate:", value=0.0435)
    sigma = st.number_input("Volatility:", value= 0.2)
    time_to_exp = st.number_input("Time to expiration [years]:", value= 1)
    annualized_dividend_yield = st.number_input("Annualized dividend yield:", value= 0.01)

    st.write("Heatmap settings:")
    sigma_min = st.number_input("Minimum volatility:", value= 0.2)
    sigma_max = st.number_input("Maximum volatility:", value= 0.8)

    if(st.checkbox("Show advanced settings: ")):
        t_steps = st.number_input("Amount of time steps:", value= 100000)
        S_steps = st.number_input("Amount of underlying asset price steps:", value= 200)
    else:
        t_steps = 100000
        S_steps = 200

sigma_vals = np.linspace(sigma_min, sigma_max, 10)
underlying_vals = np.linspace(0, 2*underlying_price, S_steps+1)

solution_grid = []
for volatility in sigma_vals:
    solution = np.array(numerical_methods.fdm_solve(strike_price, risk_free_rate, volatility, annualized_dividend_yield, time_to_exp, t_steps, underlying_price*2, S_steps, is_call))
    solution_grid.append(solution)

solution_grid = np.vstack(solution_grid)
num_cols = solution_grid.shape[1]
indices = np.linspace(0, S_steps, 10, dtype=int)
grid_subset = solution_grid[:, indices]
underlying_vals = underlying_vals[indices]
sigma_vals = [f'{vol:.2f}' for vol in sigma_vals]
underlying_vals = [f'{price:.2f}' for price in underlying_vals]


fig = go.Figure(data=go.Heatmap(
    z=grid_subset,
    x=underlying_vals,
    y=sigma_vals,
    text=grid_subset,
    texttemplate="%{text:.2f}",
    colorscale="RdYlGn_r",
    showscale=True
))

fig.update_layout(
    xaxis_title="Underlying asset price",
    yaxis_title="Volatility",
    width=600,
    height=600,
    font=dict(size=12)
)

st.plotly_chart(fig, use_container_width=True)

