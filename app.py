import streamlit as st
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# --- Sidebar Inputs ---
st.sidebar.header("Test Parameters")

mode = st.sidebar.radio(
    "Planning Mode",
    ["Estimate Sample Size", "Estimate MDE (Minimum Detectable Effect)"],
    help="Choose whether to estimate required sample size for a given uplift, or the minimum uplift detectable for a fixed sample size."
)

p_A = st.sidebar.number_input(
    "Baseline conversion rate (p_A)", min_value=0.0001, max_value=0.999, value=0.05, step=0.001,
    format="%.4f",
    help="Conversion rate for your control variant (A), e.g., 5% = 0.050"
)
thresh = st.sidebar.slider(
    "Posterior threshold (e.g., 0.95)", 0.5, 0.99, 0.95, step=0.01,
    help="Confidence level to declare a winner — usually 0.95 or 0.99"
)
desired_power = st.sidebar.slider(
    "Desired power", 0.5, 0.99, 0.8, step=0.01,
    help="Minimum acceptable power of detecting a real uplift"
)
simulations = st.sidebar.slider(
    "Simulations", 100, 2000, 300, step=100,
    help="How many test simulations to run"
)
samples = st.sidebar.slider(
    "Posterior samples", 500, 3000, 1000, step=100,
    help="How many samples to draw from each posterior distribution"
)

if mode == "Estimate Sample Size":
    uplift = st.sidebar.number_input(
        "Expected uplift (e.g., 0.10 = +10%)", min_value=0.0001, max_value=0.999, value=0.10, step=0.01,
        format="%.4f",
        help="Relative improvement expected in variant B over A"
    )
else:
    fixed_n = st.sidebar.number_input(
        "Fixed sample size per variant", min_value=100, value=10000, step=100,
        help="Fixed sample size used to determine the minimum detectable uplift."
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
        help="Prior belief in successes before the test."
    )
    beta_prior = st.sidebar.number_input(
        "Beta (prior failures)", min_value=0.0, value=1.0, step=0.1,
        help="Prior belief in failures before the test."
    )

# --- Simulate Functions ---
def simulate_power(p_A, uplift, threshold, desired_power, simulations, samples, alpha_prior, beta_prior):
    p_B = p_A * (1 + uplift)
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

def simulate_mde(p_A, threshold, desired_power, simulations, samples, alpha_prior, beta_prior, fixed_n):
    uplift = 0.001
    powers = []

    while uplift < 2.0:
        p_B = p_A * (1 + uplift)
        power_hits = 0

        for _ in range(simulations):
            conv_A = np.random.binomial(fixed_n, p_A)
            conv_B = np.random.binomial(fixed_n, p_B)

            post_A = beta(alpha_prior + conv_A, beta_prior + fixed_n - conv_A)
            post_B = beta(alpha_prior + conv_B, beta_prior + fixed_n - conv_B)

            samples_A = post_A.rvs(samples)
            samples_B = post_B.rvs(samples)

            prob_B_superior = np.mean(samples_B > samples_A)
            if prob_B_superior > threshold:
                power_hits += 1

        power = power_hits / simulations
        powers.append((uplift, power))

        if power >= desired_power:
            break
        uplift += 0.01

    return powers

# --- Run Simulation ---
st.title("🧪 Bayesian A/B Test Calculator")

if mode == "Estimate Sample Size":
    results = simulate_power(p_A, uplift, thresh, desired_power, simulations, samples, alpha_prior, beta_prior)
    x_vals, y_vals = zip(*results)

    st.subheader("📈 Sample Size Estimation")
    st.write(f"**Baseline Conversion Rate:** {p_A:.2%}")
    st.write(f"**Expected Uplift:** {uplift:.2%}")
    st.write(f"**Posterior Threshold:** {thresh:.2f}")
    st.write(f"**Target Power:** {desired_power:.0%}")
    st.write(f"**Priors Used:** Alpha = {alpha_prior:.1f}, Beta = {beta_prior:.1f}")

    if y_vals[-1] >= desired_power:
        st.success(f"✅ Estimated minimum sample size per group: {x_vals[-1]}")
    else:
        st.warning("Test did not reach desired power within simulation limits.")

    st.markdown("""
    ### 📊 What This Means
    This chart shows how sample size impacts your ability to detect the expected uplift.
    The red line shows your required power (e.g. 80%). Where the curve crosses this line is the recommended sample size.
    """)
else:
    results = simulate_mde(p_A, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, fixed_n)
    x_vals, y_vals = zip(*results)

    st.subheader("📉 Minimum Detectable Effect (MDE)")
    st.write(f"**Baseline Conversion Rate:** {p_A:.2%}")
    st.write(f"**Sample Size per Group:** {fixed_n:,}")
    st.write(f"**Posterior Threshold:** {thresh:.2f}")
    st.write(f"**Target Power:** {desired_power:.0%}")
    st.write(f"**Priors Used:** Alpha = {alpha_prior:.1f}, Beta = {beta_prior:.1f}")

    if y_vals[-1] >= desired_power:
        st.success(f"✅ Minimum detectable uplift: {x_vals[-1]:.2%}")
    else:
        st.warning("Simulation did not reach target power. Try increasing sample size or simulations.")

    st.markdown("""
    ### 📊 What This Means
    This chart shows how much uplift your test can reliably detect given your fixed sample size.
    The red line shows your required power (e.g. 80%). Where the curve crosses this line is your minimum detectable effect.
    """)

# --- Plotting ---
plt.figure(figsize=(8, 4))
plt.plot(x_vals, y_vals, marker='o')
plt.axhline(desired_power, color='red', linestyle='--', label='Target Power')
if mode == "Estimate Sample Size":
    plt.xlabel("Sample Size per Group")
else:
    plt.xlabel("Relative Uplift (MDE)")
plt.ylabel("Estimated Power")
plt.title("Power vs. " + ("Sample Size" if mode == "Estimate Sample Size" else "MDE"))
plt.grid(True)
plt.legend()
st.pyplot(plt)
