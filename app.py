import streamlit as st
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# --- Sidebar Inputs ---
st.sidebar.header("Test Parameters")
p_A = st.sidebar.number_input(
    "Baseline conversion rate (p_A)", min_value=0.01, max_value=0.20, value=0.05, step=0.001,
    format="%.3f",
    help="Conversion rate for your control variant (A), e.g., 5% = 0.050"
)
uplift = st.sidebar.number_input(
    "Expected uplift (e.g., 0.10 = +10%)", min_value=0.0, max_value=0.5, value=0.10, step=0.01,
    format="%.2f",
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

# --- Memory-efficient Simulation Function ---
def simulate_power(p_A, uplift, threshold, desired_power, simulations, samples):
    p_B = p_A * (1 + uplift)
    alpha_prior, beta_prior = 1, 1
    n = 1000
    powers = []

    while n <= 500000:
        power_hits = 0

        for _ in range(simulations):
            conv_A = np.random.binomial(n, p_A)
            conv_B = np.random.binomial(n, p_B)

            post_A = beta(alpha_prior + conv_A, beta_prior + n - conv_A)
            post_B = beta(alpha_prior + conv_B, beta_prior + n - conv_B)

            samples_A = post_A.rvs(samples)
            samples_B = post_B.rvs(samples)

            prob_B_superior = np.mean(samples_B > samples_A)
            if prob_B_superior > threshold:
                power_hits += 1

        power = power_hits / simulations
        powers.append((n, power))

        if power >= desired_power:
            break
        n += 5000

    return powers

# --- Run Simulation ---
results = simulate_power(p_A, uplift, thresh, desired_power, simulations, samples)
sample_sizes, power_values = zip(*results)

# --- Output ---
st.title("Bayesian A/B Test Power Calculator")

st.markdown("""
This app estimates the **minimum sample size per group** required to detect a given uplift in conversion rate using a Bayesian framework.
You can adjust the assumptions in the sidebar.
""")

st.write(f"**Baseline Conversion Rate:** {p_A:.2%}")
st.write(f"**Expected Uplift:** {uplift:.2%}")
st.write(f"**Posterior Threshold:** {thresh:.2f}")
st.write(f"**Target Power:** {desired_power:.0%}")

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

st.markdown("---")
