import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("""
<style>
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px;
    width: auto;
    margin: 0 auto;
}
.metric-call {
    background-color: #90ee90;
    color: black;
    margin-right: 10px;
    border-radius: 10px;
}
.metric-put {
    background-color: #ffcccb;
    color: black;
    border-radius: 10px;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    margin: 0;
}
.metric-label {
    font-size: 1rem;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

st.header('Black–Scholes–Merton Option Pricing Model')
# a black scholes merton option prcing model that value call and put options
#S: underlying stock price in USD actuals
#K: strike price in USD actuals
#T: time to maturity in years
#R: risk free rate in decimals
#sigma: volatility in decimals
@st.cache_resource
class BlackScholesMerton:
    def __init__(self, S, K, T, R, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.R = R
        self.sigma = sigma
    
    def set_S(self, S):
        self.S = S
    
    def set_K(self, K):
        self.K = K
    
    def set_T(self, T):
        self.T = T
    
    def set_R(self, R):
        self.R = R
    
    def set_sigma(self, sigma):
        self.sigma = sigma
    
    def d1_d2(self):
        d1 = (math.log(self.S / self.K) + (self.R + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)
        return d1, d2

    def value_call(self):
        d1, d2 = self.d1_d2()
        value_call = self.S * norm.cdf(d1) - self.K * math.exp(-self.R * self.T) * norm.cdf(d2)
        return value_call
    
    def greeks_call(self):
        d1, d2 = self.d1_d2()

        delta_call = norm.cdf(d1)
        gamma_call = norm.pdf(d1) / (self.S * self.sigma * math.sqrt(self.T))
        theta_call = (-self.S * norm.pdf(d1) * self.sigma / (2 * math.sqrt(self.T)) - self.R * self.K * math.exp(-self.R * self.T) * norm.cdf(d2))
        vega_call = self.S * norm.pdf(d1) * math.sqrt(self.T)
        rho_call = self.K * self.T * math.exp(-self.R * self.T) * norm.cdf(d2)

        return delta_call, gamma_call, theta_call, vega_call, rho_call
    
    def value_put(self):
        d1, d2 = self.d1_d2()
        value_put = self.K * math.exp(-self.R * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return value_put
    
    def greeks_put(self):
        d1, d2 = self.d1_d2()
        
        delta_put = norm.cdf(d1) - 1
        gamma_put = norm.pdf(d1) / (self.S * self.sigma * math.sqrt(self.T))
        theta_put = (-self.S * norm.pdf(d1) * self.sigma / (2 * math.sqrt(self.T)) + self.R * self.K * math.exp(-self.R * self.T) * norm.cdf(-d2))
        vega_put = self.S * norm.pdf(d1) * math.sqrt(self.T)
        rho_put = -self.K * self.T * math.exp(-self.R * self.T) * norm.cdf(-d2)

        return delta_put, gamma_put, theta_put, vega_put, rho_put
    
# track BSM value deltas
if 'prev_call' not in st.session_state:
    st.session_state.prev_call = 0
if 'prev_put' not in st.session_state:
    st.session_state.prev_put = 0

# sidebar: model inputs
st.sidebar.subheader("Model inputs")
S = st.sidebar.number_input("Underlying stock price (USD)", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike price (USD)", value=50.0, step=1.0)
T = st.sidebar.number_input("Time to maturity (years)", value=10.0, step=1.0)
R = st.sidebar.number_input("Risk-free rate (in decimals)", value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (in decimals)", value=0.5, step=0.01)

# Calculate new BSM values
bsm = BlackScholesMerton(S, K, T, R, sigma)
new_call = bsm.value_call()
new_put = bsm.value_put()

# Calculate differences
call_diff = new_call - st.session_state.prev_call
put_diff = new_put - st.session_state.prev_put

# Update session state
st.session_state.prev_call = new_call
st.session_state.prev_put = new_put

# option values output
col1, col2 = st.columns(2)

with col1:
    with st.container(height=125, border=True):
        st.metric(
            label="Call Option Value", 
            value=f"${round(new_call, 2)}",
            delta=f"{round(call_diff, 2)}"
        )

with col2:
    with st.container(height=125, border=True):
        st.metric(
            label="Put Option Value", 
            value=f"${round(new_put, 2)}",
            delta=f"{round(put_diff, 2)}"
        )


# sidebar: headtmap inputs
st.sidebar.subheader("Heatmap inputs")
S_sensitivity = st.sidebar.slider(
    "Range of underlying stock price",
    0.0,
    3 * S,
    (0.5*S,1.5*S)
)

sigma_sensitivity = st.sidebar.slider(
    "Range of volatilities",
    0.0,
    3 * sigma,
    (0.5*sigma,1.5*sigma)
)

def generate_sensitivity_df(bsm, S_range, sigma_range, num_points=10):
    S_values = np.linspace(S_range[1], S_range[0], num_points)
    sigma_values = np.linspace(sigma_range[0], sigma_range[1], num_points)
    
    call_values = np.zeros((num_points, num_points))
    put_values = np.zeros((num_points, num_points))
    
    for i, S in enumerate(S_values):
        for j, sigma in enumerate(sigma_values):
            # Create a new instance of BlackScholesMerton for each calculation
            temp_bsm = BlackScholesMerton(S, bsm.K, bsm.T, bsm.R, sigma)
            call_values[i, j] = temp_bsm.value_call()
            put_values[i, j] = temp_bsm.value_put()
    
    call_df = pd.DataFrame(call_values, index=S_values, columns=sigma_values)
    put_df = pd.DataFrame(put_values, index=S_values, columns=sigma_values)
    
    return call_df.round(2), put_df.round(2)

# Generate sensitivity dataframes
call_df, put_df = generate_sensitivity_df(bsm, S_sensitivity, sigma_sensitivity)

# Create heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Function to format tick labels
def format_ticks(x, pos):
    return f'{x:.2f}'

# Call Option Heatmap
custom_cmap = sns.light_palette("seagreen", as_cmap=True)
sns.heatmap(call_df, ax=ax1, cmap=custom_cmap, annot=True, fmt='.2f', 
            linewidths=0.5, cbar=True)
ax1.set_title("Call Option Value Sensitivity")
ax1.set_xlabel("Volatility")
ax1.set_ylabel("Stock Price")

# Set correct tick locations and labels
ax1.set_xticks(np.arange(len(call_df.columns)) + 0.5)
ax1.set_yticks(np.arange(len(call_df.index)) + 0.5)
ax1.set_xticklabels([f'{x:.2f}' for x in call_df.columns])
ax1.set_yticklabels([f'{y:.2f}' for y in call_df.index])

# Put Option Heatmap

sns.heatmap(put_df, ax=ax2, cmap=custom_cmap, annot=True, fmt='.2f', 
            linewidths=0.5, cbar=True)
ax2.set_title("Put Option Value Sensitivity")
ax2.set_xlabel("Volatility")
ax2.set_ylabel("Stock Price")

# Set correct tick locations and labels
ax2.set_xticks(np.arange(len(put_df.columns)) + 0.5)
ax2.set_yticks(np.arange(len(put_df.index)) + 0.5)
ax2.set_xticklabels([f'{x:.2f}' for x in put_df.columns])
ax2.set_yticklabels([f'{y:.2f}' for y in put_df.index])

# Rotate the tick labels and set their alignment
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Adjust layout to prevent clipping of tick labels
plt.tight_layout()

st.pyplot(fig)