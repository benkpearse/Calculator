import streamlit as st
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# --- Core Simulation Functions ---

def run_simulation(n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh):
    """
    Runs a single set of simulations for a given sample size and conversion rates.
    Returns the calculated power.
    """
    n_A = n
    n_B = n

    conversions_A = np.random.binomial(n_A, p_A, size=simulations)
    conversions_B = np.random.binomial(n_B, p_B, size=simulations)

    alpha_post_A = alpha_prior + conversions_A
    beta_post_A = beta_prior + n_A - conversions_A
    alpha_post_B = alpha_prior + conversions_B
    beta_post_B = beta_prior + n_B - conversions_B

    post_samples_A = beta.rvs(alpha_post_A, beta_post_A, size=(samples, simulations))
    post_samples_B = beta.rvs(alpha_post_B, beta_post_B, size=(samples, simulations))

    prob_B_better = np.mean(post_samples_B > post_samples_A, axis=0)

    power = np.mean(prob_B_better > thresh)
    return power

@st.cache_data
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
    sample_sizes = np.unique(np.logspace(2, 5, 20).astype(int))

    with st.spinner("Running simulations for sample size..."):
        for n in sample_sizes:
            power = run_simulation(n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh)
            results.append((n, power))
            if power >= desired_power:
                break
    return results

@st.cache_data
def simulate_mde(p_A, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, fixed_n):
    """
    Simulates power across a range of uplifts (MDEs) for a fixed sample size
    to find the minimum detectable effect.
    """
    results = []
    uplifts = np.linspace(0.01, 0.40, 20) 

    with st.spinner("Running simulations for MDE..."):
        for uplift in uplifts:
            p_B = p_A * (1 + uplift)
            if p_B > 1.0:
                continue
            power = run_simulation(fixed_n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh)
            results.append((uplift, power))
            if power >= desired_power:
                break
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

# --- App Body ---
st.title("Bayesian A/B Pre-Test Calculator")

results_available = False
if st.button("Run Calculation"):

    if mode == "Estimate Sample Size":
        results = simulate_power(p_A, uplift, thresh, desired_power, simulations, samples, alpha_prior, beta_prior)
        if len(results) > 1 or (len(results) == 1 and results[0][1] > 0):
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
        if len(results) > 1 or (len(results) == 1 and results[0][1] > 0):
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
    
    # --- Plotting (now inside the button click logic) ---
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
#### What is Sample Size? 👥
**Sample size** is the number of users in each group of your test (e.g., 10,000 in control, 10,000 in variant). It's the main dial you can turn to adjust your test's sensitivity.

Think of your A/B test as trying to take a picture of a distant star. The sample size (number of users) is the size of your camera lens.

A larger sample size reduces the effect of random noise, giving you a clearer, more precise measurement of each variant's true performance. This increased clarity improves your test in two related ways:

You can see smaller objects (Lower MDE)
A bigger lens (more users) lets you resolve finer details. You can now reliably detect a very faint star (a smaller effect) that would have been invisible to a smaller lens. This is why more users lead to a lower MDE.

You get a clearer picture of the object you're looking for (Higher Power)
If you're focused on one specific star (a specific expected uplift), that same big lens (more users) gives you a much better chance (higher power) of capturing a sharp, undeniable photo of it, rather than a blurry, inconclusive smudge.

In short, Power and MDE are two sides of the same coin: test sensitivity. Increasing your sample size makes your test more sensitive overall.
---
#### What is Minimum Detectable Effect (MDE)? 🔎
The **Minimum Detectable Effect (MDE)** is the smallest improvement your test can reliably detect at a given power level.

Think of it as the sensitivity of your experiment. If the true uplift from your change is smaller than the MDE, your test will likely miss it. This doesn't mean the uplift isn't real, just that your experiment isn't powerful enough to see it. Use the MDE to set realistic expectations for what your test can achieve with your available traffic.

---
#### What is Bayesian Power? 💪
**Power** answers one critical question: *"If my variant is truly better by a specific amount, what's the probability my test will actually detect it?"*

For example, 80% power means you have an 80% chance of getting a conclusive result (e.g., P(B > A) > 95%) if the real improvement matches what you expected. Running a test with low power is like trying to read in a dim room—you're likely to miss things and end up with an inconclusive result, wasting valuable traffic.

---
#### What are Priors? 🧠
**Priors** represent what you believe about the conversion rate *before* the test begins. In this model, your belief is captured by two numbers:
- **Alpha ($$\\alpha$$)**: The number of prior "successes".
- **Beta ($$\\beta$$)**: The number of prior "failures".

* **No strong belief?** Use an **uninformative prior** like `alpha = 1` and `beta = 1`. This treats all possible conversion rates as equally likely to start.
* **Have historical data?** Create an **informative prior**. If past data showed 50 conversions from 1,000 users, you'd set `alpha = 50` and `beta = 950`.

As your test collects new data, the evidence from the experiment will quickly outweigh the initial prior belief.""")
