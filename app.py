import streamlit as st
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# --- Sidebar Inputs ---
st.sidebar.header("Test Parameters")
p_A = st.sidebar.slider("Baseline conversion rate (p_A)", 0.001, 0.20, 0.05, step=0.001,
                         format="%.3f",
                         help="Conversion rate for your control variant (A), e.g., 5% = 0.050")
uplift = st.sidebar.slider("Expected uplift (e.g., 0.10 = +10%)", 0.0, 0.5, 0.10, step=0.01,
                          help="Relative improvement expected in variant B over A")
thresh = st.sidebar.slider("Posterior threshold (e.g., 0.95)", 0.5, 0.99, 0.95, step=0.01,
                           help="Confidence level to declare a winner — usually 0.95 or 0.99")
desired_power = st.sidebar.slider("Desired power", 0.5, 0.99, 0.8, step=0.01,
                                  help="Minimum acceptable probability of detecting a real uplift")
simulations = st.sidebar.slider("Simulations per n", 100, 2000, 500, step=100,
                                help="Number of test simulations to run per sample size")
samples = st.sidebar.slider("Posterior samples", 1000, 10000, 2000, step=500,
                            help="Number of samples drawn from each posterior distribution www")

# --- Vectorized Simulation Function ---
def simulate_power(p_A, uplift, threshold, desired_power, simulations, samples):
    p_B = p_A * (1 + uplift)
    alpha_prior, beta_prior = 1, 1
    n = 1000
    powers = []

    while n <= 100000:
        # Simulate conversion counts for A and B for all simulations at once
        conv_A = np.random.binomial(n, p_A, size=simulations)
        conv_B = np.random.binomial(n, p_B, size=simulations)

        # Compute posterior parameters
        alpha_A = alpha_prior + conv_A
        beta_A = beta_prior + n - conv_A
        alpha_B = alpha_prior + conv_B
        beta_B = beta_prior + n - conv_B

        # Vectorized posterior sampling: shape (simulations, samples)
        samples_A = beta.rvs(alpha_A[:, None], beta_A[:, None], size=(simulations, samples))
        samples_B = beta.rvs(alpha_B[:, None], beta_B[:, None], size=(simulations, samples))

        # Estimate the proportion of samples where B > A in each simulation
        prob_B_superior = (samples_B > samples_A).mean(axis=1)

        # Count how many simulations exceed the threshold
        power = np.mean(prob_B_superior > threshold)
        powers.append((n, power))

        if power >= desired_power:
            break
        n += 1000

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

