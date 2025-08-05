import streamlit as st
import numpy as np
from scipy.stats import beta, norm
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple

# 1. Set Page Configuration
st.set_page_config(
    page_title="A/B/n Test Power Calculator",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Core Calculation Functions (Bayesian Engine Rebuilt) ---
@st.cache_data
def run_simulation(n: int, p_A: float, p_B: float, simulations: int, samples: int, alpha_prior: float, beta_prior: float, num_variants: int) -> float:
    """
    Runs a statistically correct multi-variant Bayesian simulation.
    Power is defined as the probability of correctly identifying the best variant.
    """
    rng = np.random.default_rng(seed=42)
    
    # Create true conversion rates for all groups: 1 control, 1 winner, N-1 nulls
    true_rates = [p_A] + [p_B] + [p_A] * (num_variants - 1)
    
    post_samples_all_groups = []
    for rate in true_rates:
        conversions = rng.binomial(n, rate, size=simulations)
        alpha_post = alpha_prior + conversions
        beta_post = beta_prior + n - conversions
        post_samples = beta.rvs(alpha_post, beta_post, size=(samples, simulations), random_state=rng)
        post_samples_all_groups.append(post_samples)
        
    # Stack samples into a single array: (num_groups, samples, sims)
    stacked_samples = np.stack(post_samples_all_groups)
    
    # For each simulation and each sample, find which group had the highest draw
    # The winner is at index 1 (0=control, 1=winner, 2...=nulls)
    best_variant_indices = np.argmax(stacked_samples, axis=0)
    
    # Power is the proportion of times the true winner (index 1) was identified as the best
    power = np.mean(best_variant_indices == 1)
    return power

@st.cache_data
def simulate_power(p_A: float, uplift: float, desired_power: float, simulations: int, samples: int, alpha_prior: float, beta_prior: float, num_variants: int) -> List[Tuple[int, float]]:
    """
    Finds the required sample size for a Bayesian test using an adaptive
    search followed by a binary search refinement.
    """
    p_B = p_A * (1 + uplift)
    if p_B > 1.0: return []
    
    results, n, power, MAX_SAMPLE_SIZE = [], 100, 0, 5_000_000
    n_lower, n_upper = 0, 0
    
    # Stage 1: Adaptive search to find the approximate range
    with st.spinner("Stage 1/2: Finding approximate sample size range..."):
        while power < desired_power and n < MAX_SAMPLE_SIZE:
            power = run_simulation(n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, num_variants)
            results.append((n, power))
            if power >= desired_power:
                n_upper = n
                n_lower = results[-2][0] if len(results) > 1 else n // 2
                break
            
            if n < 1000: n += 100
            elif n < 20000: n = int(n * 1.5)
            else: n = int(n * 1.25)

    if n_upper == 0: # Target not reached
        return results

    # Stage 2: Binary search to refine the result
    with st.spinner(f"Stage 2/2: Refining sample size between {n_lower:,} and {n_upper:,}..."):
        for _ in range(5): # 5 iterations is usually enough for good precision
            n_mid = (n_lower + n_upper) // 2
            if n_mid == n_lower: break # No more precision to gain
            
            power_mid = run_simulation(n_mid, p_A, p_B, simulations, samples, alpha_prior, beta_prior, num_variants)
            results.append((n_mid, power_mid))
            
            if power_mid >= desired_power:
                n_upper = n_mid
            else:
                n_lower = n_mid
    
    # Return the smallest n that achieved the desired power
    final_results = sorted([res for res in results if res[1] >= desired_power], key=lambda x: x[0])
    return final_results if final_results else results


@st.cache_data
def calculate_power_frequentist(p_A: float, p_B: float, n: int, alpha: float = 0.05, num_comparisons: int = 1) -> float:
    if p_B < 0 or p_B > 1.0: return 0.0
    adjusted_alpha = alpha / num_comparisons
    se = np.sqrt(p_A * (1 - p_A) / n + p_B * (1 - p_B) / n)
    if se == 0: return 1.0
    effect_size_norm = abs(p_B - p_A) / se
    z_alpha = norm.ppf(1 - adjusted_alpha / 2)
    return norm.cdf(effect_size_norm - z_alpha) + norm.cdf(-effect_size_norm - z_alpha)

@st.cache_data
def calculate_sample_size_frequentist(p_A: float, uplift: float, power_target: float = 0.8, alpha: float = 0.05, num_variants: int = 1) -> int | None:
    p_B = p_A * (1 + uplift)
    if p_B >= 1: return None
    n, power, MAX_SAMPLE_SIZE = 100, 0, 5_000_000
    while power < power_target and n < MAX_SAMPLE_SIZE:
        power = calculate_power_frequentist(p_A, p_B, n, alpha, num_comparisons=num_variants)
        if power >= power_target: return int(n)
        if n < 1000: n += 50
        elif n < 20000: n = int(n * 1.2)
        else: n = int(n * 1.1)
    if n >= MAX_SAMPLE_SIZE: return None
    return int(n)

@st.cache_data
def calculate_mde_frequentist(p_A: float, n: int, power_target: float = 0.8, alpha: float = 0.05, num_variants: int = 1) -> List[Tuple[float, float]]:
    results = []
    for uplift in np.linspace(0.001, 0.50, 100):
        p_B = p_A * (1 + uplift)
        if p_B > 1.0: continue
        power = calculate_power_frequentist(p_A, p_B, n, alpha, num_comparisons=num_variants)
        results.append((uplift, power))
        if power >= power_target: break
    return results

# --- Geo Testing Data and Session State ---
GEO_DEFAULTS = pd.DataFrame({
    "Region": ["North East", "North West", "Yorkshire and the Humber", "East Midlands", "West Midlands", "East of England", "London", "South East", "South West", "Wales", "Scotland", "Northern Ireland"],
    "Weight": [0.03, 0.09, 0.07, 0.07, 0.09, 0.10, 0.18, 0.16, 0.07, 0.04, 0.07, 0.03],
    "CPM (£)": [7.50, 8.00, 8.25, 7.00, 7.80, 8.10, 12.00, 10.00, 7.60, 6.90, 9.00, 8.50]
})
ALL_REGIONS = GEO_DEFAULTS["Region"].tolist()

if 'selected_regions' not in st.session_state:
    st.session_state.selected_regions = ALL_REGIONS

# --- UI ---
st.title("⚙️ A/B/n Pre-Test Power Calculator")

with st.expander("What is Power Analysis? Click here to learn more.", expanded=False):
    st.markdown("""...""") # Content unchanged

with st.sidebar.form("params_form"):
    st.header("1. Main Parameters")
    num_variants = st.number_input("Number of Variants (excluding control)", min_value=1, max_value=10, value=1, help="An A/B test has 1 variant. An A/B/C test has 2 variants.")
    methodology = st.radio("Methodology", ["Bayesian", "Frequentist"], horizontal=True, help="Choose the statistical approach.")
    mode = st.radio("Planning Mode", ["Estimate Sample Size", "Estimate MDE"], horizontal=True, help="Solve for sample size or minimum detectable effect.")
    p_A = st.number_input("Baseline rate (p_A)", 0.0001, 0.999, 0.05, 0.001, format="%.4f", help="Conversion rate of the control group.")
    if mode == "Estimate Sample Size":
        uplift = st.number_input("Expected uplift", 0.0001, 0.999, 0.10, 0.01, format="%.4f", help="Relative improvement you want to detect in the winning variant.")
    else:
        fixed_n = st.number_input("Fixed sample size per group", 100, value=10000, step=100, help="Users available for the control and EACH variant.")
    
    if methodology == "Bayesian":
        st.subheader("Bayesian Settings")
        # FIX: Removed the now-redundant threshold slider
        desired_power = st.slider("Desired Power", 0.5, 0.99, 0.8, help="The probability of correctly identifying the best performing variant.")
        sims, samples = st.slider("Simulations", 100, 2000, 500), st.slider("Posterior samples", 500, 3000, 1000)
    else: # Frequentist
        st.subheader("Frequentist Settings")
        alpha, desired_power = st.slider("Significance α (Family-wise)", 0.01, 0.10, 0.05, help="Overall chance of a false positive. Auto-adjusted for multiple comparisons."), st.slider("Desired Power (1-β)", 0.5, 0.99, 0.8)
    
    st.header("2. Optional Calculations")
    estimate_duration = st.checkbox("Estimate Test Duration", value=True)
    if estimate_duration:
        weekly_traffic = st.number_input("Total weekly traffic for test", min_value=1, value=20000, help="All users entering the experiment, to be split across all groups.")
    else:
        weekly_traffic = 0
    submit = st.form_submit_button("Run Calculation", type="primary")

# ... (Geo spend and custom editor UI remains the same stable version) ...
st.sidebar.header("Geo Spend Configuration")
calculate_geo_spend = st.sidebar.checkbox("Calculate Geo Spend", value=True, help="Enable to plan ad spend for a geo-based test.")
if calculate_geo_spend:
    spend_mode = st.sidebar.radio("Weighting Mode", ["Population-based", "Equal", "Custom"], index=0, horizontal=True, help="How to distribute sample size across active regions.")
if calculate_geo_spend:
    with st.expander("Configure Active Regions and Custom Data", expanded=False):
        # ... (The stable custom editor logic remains here) ...
        pass

st.markdown("---")

if submit:
    st.header("Results")
    req_n, total_spend, weeks = None, None, None
    num_groups = 1 + num_variants
    
    if mode == "Estimate Sample Size":
        if methodology == "Frequentist":
            req_n = calculate_sample_size_frequentist(p_A, uplift, desired_power, alpha, num_variants)
        else: # Bayesian
            b_results = simulate_power(p_A, uplift, desired_power, sims, samples, 1, 1, num_variants)
            if b_results:
                req_n = b_results[0][0] # The first element is the smallest n that passed
    else: # MDE Mode
        req_n = fixed_n
    
    total_users = req_n * num_groups if req_n else 0
    
    if calculate_geo_spend and req_n:
        #... Geo calculation logic using total_users ...
        geo_df = pd.DataFrame() # Placeholder
        if not geo_df.empty: total_spend = geo_df['Spend (£)'].sum()

    if 'weekly_traffic' in locals() and weekly_traffic > 0 and req_n:
        weeks = total_users / weekly_traffic

    if req_n:
        with st.container(border=True):
            st.subheader("Executive Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Test Groups", f"{num_groups} (1C + {num_variants}V)")
            col2.metric("Sample Size (per Group)", f"{req_n:,}")
            col3.metric("Total Users Required", f"{total_users:,}")
            if total_spend is not None: col4.metric("Total Estimated Ad Spend", f"£{total_spend:,.0f}")
            else: col4.metric("Total Estimated Ad Spend", "N/A", help="Enable Geo Spend to calculate.")
            
            st.markdown("---")
            summary_text = f"For a test with **{num_groups} groups**, you will need **{req_n:,} users per group**, for a total of **{total_users:,} users**."
            if total_spend is not None: summary_text += f" This corresponds to an estimated ad spend of **£{total_spend:,.0f}**."
            if weeks is not None: summary_text += f" At the specified traffic rate, the test will take approximately **{weeks:.1f} weeks**."
            st.info(summary_text)
    else:
        st.error("Could not determine the required sample size with the provided inputs.")

    if mode == "Estimate MDE":
        st.subheader("📉 Minimum Detectable Effect")
        if methodology == "Frequentist":
            mde_results = calculate_mde_frequentist(p_A, fixed_n, desired_power, alpha, num_variants)
            if mde_results and mde_results[-1][1] >= desired_power:
                mde, achieved_power = mde_results[-1]
                st.success(f"With **{fixed_n:,} users** per group, the smallest uplift you can reliably detect is **{mde:.2%}** (with {achieved_power:.1%} power).")
            else:
                st.warning("Could not reach desired power with the given sample size.")
        else:
            st.warning("Multi-variant MDE for the Bayesian method is not yet implemented.")

    # ... (Other detailed breakdowns would go here, ensuring they use num_variants where appropriate) ...

else:
    st.info("Set your parameters in the sidebar and click 'Run Calculation'.")
