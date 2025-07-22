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

# --- Run Simulation ---
st.title("Bayesian A/B Pre Test Calculator")

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

# --- Conceptual Explanation ---
st.markdown("""
<details>
<summary><strong>ℹ️ What is Minimum Detectable Effect (MDE)?</strong></summary>

**Minimum Detectable Effect (MDE)** tells you the smallest improvement (uplift) your test is likely to detect with a given amount of data.

If your true uplift is smaller than the MDE, you probably won’t detect it — not because it's not real, but because your test isn't sensitive enough.

Use MDE to set realistic expectations: if your MDE is 5%, don’t expect to reliably detect a 2% improvement.

</details>

<details>
<summary><strong>ℹ️ What Does Power Mean in Bayesian A/B Testing?</strong></summary>

Bayesian power answers this question:

> **If the improvement is real, how often will my test be confident enough to detect it?**

We define “confident enough” as your posterior probability threshold (e.g., P(B > A) > 0.95).

So if you expect a 10% uplift and run 300 tests with that uplift, power is the percent of those that correctly conclude B is better than A.

This helps you decide how much data is needed before starting a real test.

</details>

<details>
<summary><strong>ℹ️ About Priors in Bayesian A/B Testing</strong></summary>

Priors represent your prior beliefs about the conversion rate before running the test.

A **Beta prior** is defined by two parameters:
- **Alpha** = prior successes
- **Beta** = prior failures

If you have **no strong prior belief**, use `alpha = 1`, `beta = 1` — this is called a uniform (or uninformative) prior.

If you **have historical data**, you can encode it using:
- Prior conversion rate (e.g., 0.05)
- Prior sample size (e.g., 1000)

These get translated into alpha and beta by:
- `alpha = conversion rate × sample size`
- `beta = (1 - conversion rate) × sample size`

Over time, you can build better priors by accumulating test outcomes in similar contexts (e.g., same site, funnel, device). This makes your tests more data-efficient.

</details>
""", unsafe_allow_html=True)
# --- Time-Based Test Planning ---
st.markdown("---")
st.header("⏱️ Time-Based Test Planning")

st.markdown("""
Use this section to estimate how long your test will take, or what uplift you can detect based on how long you're able to run the test.
""")

# Input: Total expected traffic per week (before splitting)
weekly_traffic = st.number_input(
    "Estimated total weekly traffic (users hitting the test)",
    min_value=100,
    value=10000,
    step=100
)

# Option: Fixed test duration
test_duration_weeks = st.number_input(
    "Test duration (weeks)",
    min_value=1,
    value=3,
    step=1
)

# Option A: Estimate duration from sample size
if mode == "Estimate Sample Size":
    required_sample_size = None
    try:
        required_sample_size = x_vals[-1]  # already computed from simulate_power
        users_per_week_per_variant = weekly_traffic / 2  # 50/50 split
        estimated_weeks = required_sample_size / users_per_week_per_variant

        st.subheader("🧮 Duration Estimate")
        st.success(f"To collect {required_sample_size:,} users per variant, you need approx. **{estimated_weeks:.1f} weeks** of testing.")
    except:
        pass

# Option B: Estimate max sample size from time cap
max_total_users = weekly_traffic * test_duration_weeks
max_per_variant = max_total_users // 2

st.subheader("🎯 Max Sample Size From Time Cap")
st.write(f"With {weekly_traffic:,} users/week and a test duration of {test_duration_weeks} weeks:")
st.info(f"➡️ You can test up to **{max_per_variant:,} users per variant**")

# Optional: Link directly to MDE calculator flow
if mode == "Estimate MDE":
    st.markdown("---")
    st.subheader("🔍 Re-estimate MDE using time-limited sample size")
    if st.button("Recalculate MDE with max sample size"):
        time_limited_results = simulate_mde(
            p_A, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, max_per_variant
        )
        u_vals, p_vals = zip(*time_limited_results)

        if p_vals[-1] >= desired_power:
            st.success(f"✅ Minimum detectable uplift (within time/traffic limit): {u_vals[-1]:.2%}")
        else:
            st.warning("Could not reach target power with this sample size. Try increasing test duration or traffic.")

        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(u_vals, p_vals, marker='o')
        plt.axhline(desired_power, color='red', linestyle='--', label='Target Power')
        plt.xlabel("Uplift")
        plt.ylabel("Power")
        plt.title("Power vs Uplift with Time-Limited Sample Size")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

        # Summary Box
        st.markdown("### 📝 Summary")
        st.write(f"- **Traffic/week**: {weekly_traffic:,} users")
        st.write(f"- **Test duration**: {test_duration_weeks} weeks")
        st.write(f"- **Max users per variant**: {max_per_variant:,}")
        st.write(f"- **Threshold**: {thresh:.2f}, **Target Power**: {desired_power:.0%}")
        st.write(f"- **Minimum detectable uplift**: {u_vals[-1]:.2%}")

st.caption("You can plug this value into the Minimum Detectable Effect (MDE) estimator to see what uplift you'd be able to detect within this traffic/time budget.")
