import streamlit as st
import numpy as np
from scipy.stats import beta
from numba import njit
import matplotlib.pyplot as plt

# --- Sidebar Inputs ---
st.sidebar.header("Test Parameters")
p_A = st.sidebar.number_input(
    "Baseline conversion rate (p_A)", min_value=0.01, max_value=0.99, value=0.05, step=0.001,
    format="%.3f",
    help="Conversion rate for your control variant (A), e.g., 5% = 0.050"
)
uplift = st.sidebar.number_input(
    "Expected uplift (e.g., 0.10 = +10%)", min_value=0.0, max_value=0.99, value=0.10, step=0.01,
    format="%.3f",
    help="Relative improvement expected in variant B over A"
)
thresh = st.sidebar.slider(
    "Posterior threshold (e.g., 0.95)", 0.5, 0.99, 0.95, step=0.01,
    help="Confidence level to declare a winner — usually 0.95 or 0.99"
)
desired_power = st.sidebar.slider(
    "Desired power", 0.5, 0.99, 0.8, step=0.01,
    help="Minimum acceptable probability of detecting a real uplift"
)
simulations = st.sidebar.slider(
    "Simulations per n", 100, 2000, 300, step=100,
    help="Number of test simulations to run per sample size"
)
samples = st.sidebar.slider(
    "Posterior samples", 500, 3000, 1000, step=100,
    help="Number of samples drawn from each posterior distribution test"
)

# --- Optional Priors ---
st.sidebar.markdown("---")
st.sidebar.subheader("Optional Prior Beliefs")

use_auto_prior = st.sidebar.checkbox(
    "Auto-calculate priors from historical data",
    help="Check this to calculate priors based on a past conversion rate and sample size."
)

if use_auto_prior:
    hist_cr = st.sidebar.number_input(
        "Historical conversion rate (0.05 = 5%)", min_value=0.0, max_value=1.0, value=0.05, step=0.001,
        format="%.3f",
        help="Observed conversion rate from your historical data."
    )
    hist_n = st.sidebar.number_input(
        "Historical sample size", min_value=1, value=1000, step=1,
        help="Number of observations (users) in historical data."
    )
    alpha_prior = hist_cr * hist_n
    beta_prior = (1 - hist_cr) * hist_n
else:
    alpha_prior = st.sidebar.number_input(
        "Alpha (prior successes)", min_value=0.0, value=1.0, step=0.1,
        help="Represents prior belief in success count before the test. Higher values add more weight to your prior knowledge."
    )
    beta_prior = st.sidebar.number_input(
        "Beta (prior failures)", min_value=0.0, value=1.0, step=0.1,
        help="Represents prior belief in failure count before the test. Adjust this if you have historical data."
    )

# --- Simulation Function using Numba ---
@njit
def simulate_single_trial(n, p_A, p_B, samples, alpha_prior, beta_prior, threshold):
    conv_A = np.random.binomial(n, p_A)
    conv_B = np.random.binomial(n, p_B)

    alpha_A = alpha_prior + conv_A
    beta_A = beta_prior + n - conv_A
    alpha_B = alpha_prior + conv_B
    beta_B = beta_prior + n - conv_B

    wins = 0
    for _ in range(samples):
        sample_A = np.random.beta(alpha_A, beta_A)
        sample_B = np.random.beta(alpha_B, beta_B)
        if sample_B > sample_A:
            wins += 1

    return wins / samples > threshold

@njit
def simulate_power(p_A, uplift, threshold, desired_power, simulations, samples, alpha_prior, beta_prior):
    p_B = p_A * (1 + uplift)
    n = 1000
    results = []

    while n <= 500000:
        wins = 0
        for _ in range(simulations):
            if simulate_single_trial(n, p_A, p_B, samples, alpha_prior, beta_prior, threshold):
                wins += 1
        power = wins / simulations
        results.append((n, power))
        if power >= desired_power:
            break
        n += 5000

    return results

# --- Run Simulation ---
results = simulate_power(p_A, uplift, thresh, desired_power, simulations, samples, alpha_prior, beta_prior)
sample_sizes, power_values = zip(*results)

# --- Output ---
st.title("Bayesian A/B Test Power Calculator")

st.markdown("""
This app estimates the **minimum sample size per group** required to detect a given uplift in conversion rate using a Bayesian framework.
You can adjust the assumptions in the sidebar.

---
**ℹ️ About Priors:**
Priors represent your prior beliefs about the conversion rate before running the test. 
- A Beta prior is defined by two parameters: `alpha` (prior successes) and `beta` (prior failures).
- If you have no strong prior belief, use `alpha = 1`, `beta = 1` — this is called a uniform prior.
- If you have historical data, you can encode it here to improve inference efficiency.
- Now you can optionally auto-calculate these priors using a historical conversion rate and sample size.
""")

st.write(f"**Baseline Conversion Rate:** {p_A:.2%}")
st.write(f"**Expected Uplift:** {uplift:.2%}")
st.write(f"**Posterior Threshold:** {thresh:.2f}")
st.write(f"**Target Power:** {desired_power:.0%}")
st.write(f"**Priors Used:** Alpha = {alpha_prior:.1f}, Beta = {beta_prior:.1f}")

if power_values[-1] >= desired_power:
    st.success(f"✅ Estimated minimum sample size per group: {sample_sizes[-1]}")
else:
    st.warning("Test did not reach desired power within simulation limits.")

# --- Plotting ---
plt.figure(figsize=(8, 4))
plt.plot(sample_sizes, power_values, marker='o')
plt.axhline(desired_power, color='red', linestyle='--', label='Target Power')
plt.xlabel("Sample Size per Group")
plt.ylabel("Estimated Power")
plt.title("Power Curve")
plt.grid(True)
plt.legend()
st.pyplot(plt)
