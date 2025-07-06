import streamlit as st
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# --- App Title and Navigation ---
st.set_page_config(
    page_title="Bayesian Test Suite",
    page_icon="/mnt/data/b1b01ac3-61de-4374-84ce-d254142ebdac.png",
    layout="wide"
)
st.image("/mnt/data/b1b01ac3-61de-4374-84ce-d254142ebdac.png", width=50)
st.title("🧪 Bayesian A/B Test Power Calculator")

st.markdown("""
This tool estimates the **minimum sample size per group** required to detect a given uplift in conversion rate using a Bayesian framework.

🔍 **When to use:**
- During planning: to determine the sample size needed based on your expected uplift.
- During testing: to evaluate if your current sample is large enough.
- After testing: to validate whether you had enough power to make conclusions.

📊 Adjust parameters and priors in the sidebar.
""")

# --- Sidebar Inputs ---
st.sidebar.header("🛠️ Test Parameters")
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
st.sidebar.subheader("📚 Optional Prior Beliefs")

st.sidebar.markdown("""
Use prior beliefs to incorporate historical data into your test.

- **Alpha** represents prior conversions (successes).
- **Beta** represents prior non-conversions (failures).
- You can manually enter them or auto-calculate from historical data.
""")

use_auto_prior = st.sidebar.checkbox(
    "Auto-calculate priors from historical data",
    help="Use a historical conversion rate and sample size to generate prior beliefs."
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
        help="Prior belief in conversions. Use values >1 if you have historical knowledge."
    )
    beta_prior = st.sidebar.number_input(
        "Beta (prior failures)", min_value=0.0, value=1.0, step=0.1,
        help="Prior belief in non-conversions."
    )

# --- Explanation of Priors ---
st.markdown("""
---

📘 **About Priors**

Bayesian power analysis allows you to incorporate existing knowledge using priors.

- Use **Alpha** and **Beta** to reflect prior beliefs from historical data.
- You can **auto-calculate** them from past conversion rates and sample sizes.
- If you’re unsure, the default of `Alpha = 1`, `Beta = 1` is a non-informative prior (neutral).
""")

# --- Power Calculation Function ---
def simulate_power(p_A, uplift, threshold, desired_power, simulations, samples, alpha_prior, beta_prior):
    p_B = p_A * (1 + uplift)
    n = 1000
    results = []

    while n <= 500000:
        wins = 0
        for _ in range(simulations):
            conv_A = np.random.binomial(n, p_A)
            conv_B = np.random.binomial(n, p_B)

            alpha_A = alpha_prior + conv_A
            beta_A = beta_prior + n - conv_A
            alpha_B = alpha_prior + conv_B
            beta_B = beta_prior + n - conv_B

            samples_A = np.random.beta(alpha_A, beta_A, samples)
            samples_B = np.random.beta(alpha_B, beta_B, samples)
            prob_B_superior = np.mean(samples_B > samples_A)

            if prob_B_superior > threshold:
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
st.subheader("📈 Results")
st.write(f"**Baseline Conversion Rate:** {p_A:.2%}")
st.write(f"**Expected Uplift:** {uplift:.2%}")
st.write(f"**Posterior Threshold:** {thresh:.2f}")
st.write(f"**Target Power:** {desired_power:.0%}")
st.write(f"**Priors Used:** Alpha = {alpha_prior:.1f}, Beta = {beta_prior:.1f}")

if power_values[-1] >= desired_power:
    st.success(f"✅ Estimated minimum sample size per group: {sample_sizes[-1]}")
else:
    st.warning("⚠️ Test did not reach desired power within simulation limits.")

# --- CSV Export ---
st.markdown("### 📂 Export")
st.caption("Download the power curve data for your own reporting or analysis.")
st.download_button(
    label="📏 Download CSV",
    data=f"Sample Size,Power\n" + "\n".join([f"{n},{p}" for n, p in results]),
    file_name="bayesian_power_curve.csv",
    mime="text/csv",
)

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

# --- Educational Tip ---
st.markdown("""
<details>
<summary><strong>ℹ️ What this chart means</strong></summary>

- The **power curve** shows how your ability to detect a real uplift increases as sample size grows.
- A power of 80% (red line) is often considered the minimum acceptable level.
- The vertical position where the curve crosses this line is your required sample size.

This helps prevent **underpowered tests**, which might miss real effects.

</details>
""", unsafe_allow_html=True)
