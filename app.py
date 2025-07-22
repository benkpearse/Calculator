import streamlit as st
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import pandas as pd

# --- Core Simulation Functions (MISSING FROM ORIGINAL CODE) ---

@st.cache_data
def run_simulation(n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh):
    """
    Runs a single set of simulations for a given sample size and conversion rates.
    Returns the calculated power.
    """
    n_A = n
    n_B = n

    # Generate synthetic data from simulations
    conversions_A = np.random.binomial(n_A, p_A, size=simulations)
    conversions_B = np.random.binomial(n_B, p_B, size=simulations)

    # Calculate posteriors for each simulation
    alpha_post_A = alpha_prior + conversions_A
    beta_post_A = beta_prior + n_A - conversions_A
    alpha_post_B = alpha_prior + conversions_B
    beta_post_B = beta_prior + n_B - conversions_B

    # Draw samples from posterior distributions
    post_samples_A = beta.rvs(alpha_post_A, beta_post_A, size=(samples, simulations))
    post_samples_B = beta.rvs(alpha_post_B, beta_post_B, size=(samples, simulations))

    # Calculate probability of B > A for each simulation
    prob_B_better = np.mean(post_samples_B > post_samples_A, axis=0)

    # Power is the proportion of simulations where we correctly detect the effect
    power = np.mean(prob_B_better > thresh)
    return power


def simulate_power(p_A, uplift, thresh, desired_power, simulations, samples, alpha_prior, beta_prior):
    """
    Simulates power across a range of sample sizes to find the minimum
    sample size required to achieve the desired power.
    """
    p_B = p_A * (1 + uplift)
    if p_B > 1.0:
        st.error(f"Error: Uplift of {uplift:.2%} on baseline {p_A:.2%} results in a conversion rate > 100%. Please lower the uplift or baseline.")
        return [(0, 0)]

    results = []
    # Start with a reasonable sample size and increase it
    sample_sizes = np.unique(np.logspace(2, 5, 20).astype(int)) # e.g., from 100 to 100,000

    with st.spinner("Running simulations for sample size..."):
        for n in sample_sizes:
            power = run_simulation(n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh)
            results.append((n, power))
            if power >= desired_power:
                break # Stop once we've reached the target power
    return results


def simulate_mde(p_A, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, fixed_n):
    """
    Simulates power across a range of uplifts (MDEs) for a fixed sample size
    to find the minimum detectable effect.
    """
    results = []
    # Test a range of potential uplifts
    uplifts = np.linspace(0.01, 0.40, 20) # e.g., from 1% to 40% uplift

    with st.spinner("Running simulations for MDE..."):
        for uplift in uplifts:
            p_B = p_A * (1 + uplift)
            if p_B > 1.0:
                continue # Skip invalid conversion rates
            power = run_simulation(fixed_n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh)
            results.append((uplift, power))
            if power >= desired_power:
                break # Stop once we've found an uplift that meets the power requirement
    return results


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
st.title("Bayesian A/B Pre-Test Calculator")

# Initialize placeholder
x_vals, y_vals = [0], [0]
results_available = False

if st.button("Run Calculation"):

    if mode == "Estimate Sample Size":
        results = simulate_power(p_A, uplift, thresh, desired_power, simulations, samples, alpha_prior, beta_prior)
        if len(results) > 1 or (len(results) == 1 and results[0][1] > 0): # Check if simulation ran successfully
            x_vals, y_vals = zip(*results)
            results_available = True
    
            st.subheader("📈 Sample Size Estimation")
            st.write(f"**Baseline Conversion Rate:** {p_A:.2%}")
            st.write(f"**Expected Uplift:** {uplift:.2%}")
            st.write(f"**Posterior Threshold:** {thresh:.2f}")
            st.write(f"**Target Power:** {desired_power:.0%}")
            st.write(f"**Priors Used:** Alpha = {alpha_prior:.1f}, Beta = {beta_prior:.1f}")
    
            if y_vals[-1] >= desired_power:
                st.success(f"✅ Estimated minimum sample size per group: **{x_vals[-1]:,}**")
            else:
                st.warning("Test did not reach desired power within simulation limits. Try increasing the max sample size range or lowering power target.")
    
            st.markdown("""
            ### 📊 What This Means
            This chart shows how sample size impacts your ability to detect the expected uplift.
            The red line shows your required power (e.g. 80%). Where the curve crosses this line is the recommended sample size.
            """)
    else: # Estimate MDE Mode
        results = simulate_mde(p_A, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, fixed_n)
        if len(results) > 1 or (len(results) == 1 and results[0][1] > 0): # Check if simulation ran successfully
            x_vals, y_vals = zip(*results)
            results_available = True
    
            st.subheader("📉 Minimum Detectable Effect (MDE)")
            st.write(f"**Baseline Conversion Rate:** {p_A:.2%}")
            st.write(f"**Sample Size per Group:** {fixed_n:,}")
            st.write(f"**Posterior Threshold:** {thresh:.2f}")
            st.write(f"**Target Power:** {desired_power:.0%}")
            st.write(f"**Priors Used:** Alpha = {alpha_prior:.1f}, Beta = {beta_prior:.1f}")
    
            if y_vals[-1] >= desired_power:
                st.success(f"✅ Minimum detectable relative uplift: **{x_vals[-1]:.2%}**")
            else:
                st.warning("Simulation did not reach target power. Try increasing sample size or simulations, or check if the uplift range is realistic.")
    
            st.markdown("""
            ### 📊 What This Means
            This chart shows how much uplift your test can reliably detect given your fixed sample size.
            The red line shows your required power (e.g. 80%). Where the curve crosses this line is your minimum detectable effect.
            """)
    
    # --- Plotting ---
    if results_available:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_vals, y_vals, marker='o', label='Estimated Power')
        ax.axhline(desired_power, color='red', linestyle='--', label='Target Power')
        if mode == "Estimate Sample Size":
            ax.set_xlabel("Sample Size per Group")
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
        else:
            ax.set_xlabel("Relative Uplift (MDE)")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
        ax.set_ylabel("Estimated Power")
        ax.set_title("Power vs. " + ("Sample Size" if mode == "Estimate Sample Size" else "MDE"))
        ax.grid(True, which="both", ls="--", c='0.7')
        ax.legend()
        st.pyplot(fig)


# --- Conceptual Explanation ---
st.markdown("---")
with st.expander("ℹ️ Learn about the concepts used in this calculator"):
    st.markdown("""
    #### What is Minimum Detectable Effect (MDE)?
    **Minimum Detectable Effect (MDE)** tells you the smallest improvement (uplift) your test is likely to detect with a given amount of data and a desired level of power. If the true uplift is smaller than the MDE, you probably won’t find a statistically significant result, not because the uplift isn't real, but because your test isn't sensitive enough. Use MDE to set realistic expectations: if your calculated MDE is 5%, don’t expect to reliably detect a 2% improvement.

    ---
    #### What Does Power Mean in Bayesian A/B Testing?
    In this context, Bayesian power answers the question: **"If the true uplift is X%, what is the probability that our test will correctly conclude that the variant is better than the control?"** We define "correctly conclude" as the posterior probability `P(B > A)` exceeding our chosen confidence threshold (e.g., 95%). For example, 80% power means that if the true uplift really exists, 80% of the time we run this test, we'll get a significant result. This helps you avoid launching tests that are underpowered and likely to fail from the start.

    ---
    #### About Priors in Bayesian A/B Testing
    Priors represent your beliefs about the conversion rate *before* running the test. A **Beta prior** is defined by two shape parameters:
    - **Alpha ($$\\alpha$$)**: Represents prior successes.
    - **Beta ($$\\beta$$)**: Represents prior failures.

    * **Uninformative Prior**: If you have no strong prior belief, use `alpha = 1` and `beta = 1`. This is a uniform prior, meaning all conversion rates are considered equally likely initially.
    * **Informative Prior**: If you have historical data, you can encode it. For `1,000` historical observations and a `5%` conversion rate, your priors would be:
        - `alpha = 1000 * 0.05 = 50`
        - `beta = 1000 * (1 - 0.05) = 950`

    Using good priors makes your tests more data-efficient.
    """)
```
