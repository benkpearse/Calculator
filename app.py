import streamlit as st
import numpy as np
from scipy.stats import beta, norm
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple

# 1. Set Page Configuration
st.set_page_config(
    page_title="Power Calculator | Bayesian Toolkit",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Core Calculation Functions (Unchanged) ---

@st.cache_data
def run_simulation(n: int, p_A: float, p_B: float, simulations: int, samples: int, alpha_prior: float, beta_prior: float, thresh: float) -> float:
    """Runs a batch of Bayesian A/B test simulations for a given sample size."""
    n_A = n
    n_B = n
    rng = np.random.default_rng(seed=42)
    conversions_A = rng.binomial(n_A, p_A, size=simulations)
    conversions_B = rng.binomial(n_B, p_B, size=simulations)
    
    alpha_post_A = alpha_prior + conversions_A
    beta_post_A = beta_prior + n_A - conversions_A
    alpha_post_B = alpha_prior + conversions_B
    beta_post_B = beta_prior + n_B - conversions_B
    
    post_samples_A = beta.rvs(alpha_post_A, beta_post_A, size=(samples, simulations), random_state=rng)
    post_samples_B = beta.rvs(alpha_post_B, beta_post_B, size=(samples, simulations), random_state=rng)
    
    prob_B_better = np.mean(post_samples_B > post_samples_A, axis=0)
    power = np.mean(prob_B_better > thresh)
    return power

@st.cache_data
def simulate_power(p_A: float, uplift: float, thresh: float, desired_power: float, simulations: int, samples: int, alpha_prior: float, beta_prior: float) -> List[Tuple[int, float]]:
    """Finds the required sample size for a Bayesian test to achieve desired power."""
    p_B = p_A * (1 + uplift)
    if p_B > 1.0: return []
    
    results = []
    n = 100
    power = 0
    MAX_SAMPLE_SIZE = 5_000_000
    with st.spinner("Searching for required sample size..."):
        while power < desired_power and n < MAX_SAMPLE_SIZE:
            power = run_simulation(n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh)
            results.append((n, power))
            if power >= desired_power: break
            
            if n < 1000: n += 100
            elif n < 20000: n = int(n * 1.5)
            else: n = int(n * 1.25)
    return results

@st.cache_data
def simulate_mde(p_A: float, thresh: float, desired_power: float, simulations: int, samples: int, alpha_prior: float, beta_prior: float, fixed_n: int) -> List[Tuple[float, float]]:
    """Finds the Minimum Detectable Effect for a Bayesian test with a fixed sample size."""
    results = []
    uplifts = np.linspace(0.01, 0.50, 20)
    with st.spinner("Running simulations for MDE..."):
        for uplift in uplifts:
            p_B = p_A * (1 + uplift)
            if p_B > 1.0: continue
            
            power = run_simulation(fixed_n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh)
            results.append((uplift, power))
            if power >= desired_power: break
    return results

@st.cache_data
def calculate_power_frequentist(p_A: float, p_B: float, n: int, alpha: float = 0.05) -> float:
    """Calculates the strict two-sided power for a frequentist A/B test."""
    if p_B < 0 or p_B > 1.0:
        return 0.0

    se = np.sqrt(p_A * (1 - p_A) / n + p_B * (1 - p_B) / n)
    if se == 0:
        return 1.0

    effect_size_norm = abs(p_B - p_A) / se
    z_alpha = norm.ppf(1 - alpha / 2)
    
    power = norm.cdf(effect_size_norm - z_alpha) + norm.cdf(-effect_size_norm - z_alpha)
    return power

@st.cache_data
def calculate_sample_size_frequentist(p_A: float, uplift: float, power_target: float = 0.8, alpha: float = 0.05) -> int | None:
    """Calculates required sample size by iteratively searching for 'n'."""
    p_B = p_A * (1 + uplift)
    if p_B >= 1:
        return None

    n = 100
    power = 0
    MAX_SAMPLE_SIZE = 5_000_000

    while power < power_target and n < MAX_SAMPLE_SIZE:
        power = calculate_power_frequentist(p_A, p_B, n, alpha)
        
        if power >= power_target:
            return int(n)

        if n < 1000: n += 50
        elif n < 20000: n = int(n * 1.2)
        else: n = int(n * 1.1)
    
    if n >= MAX_SAMPLE_SIZE: return None
    return int(n)


@st.cache_data
def calculate_mde_frequentist(p_A: float, n: int, power_target: float = 0.8, alpha: float = 0.05) -> List[Tuple[float, float]]:
    """Calculates MDE by iteratively searching for the uplift that meets the target power."""
    results = []
    for uplift in np.linspace(0.001, 0.50, 100):
        p_B = p_A * (1 + uplift)
        if p_B > 1.0: continue
        power = calculate_power_frequentist(p_A, p_B, n, alpha)
        results.append((uplift, power))
        if power >= power_target: break
    return results

# --- Geo Testing Data ---
# NEW: Create a master DataFrame for all default geo data
GEO_DEFAULTS = pd.DataFrame({
    "Region": ["North East", "North West", "Yorkshire and the Humber", "East Midlands", "West Midlands", "East of England", "London", "South East", "South West", "Wales", "Scotland", "Northern Ireland"],
    "Weight": [0.03, 0.09, 0.07, 0.07, 0.09, 0.10, 0.18, 0.16, 0.07, 0.04, 0.07, 0.03], # Population Weights
    "CPM (£)": [7.50, 8.00, 8.25, 7.00, 7.80, 8.10, 12.00, 10.00, 7.60, 6.90, 9.00, 8.50]
})

# --- UI ---
st.title("⚙️ Pre-Test Power Calculator")
st.markdown("Choose Bayesian (simulation) or Frequentist (formula) to estimate required sample size, then optionally calculate Geo Ad Spend.")

# Sidebar form
with st.sidebar.form("params_form"):
    st.header("1. Method & Inputs")
    methodology = st.radio("Methodology", ["Bayesian", "Frequentist"], horizontal=True, help="Choose the statistical approach for the calculation.")
    mode = st.radio("Planning Mode", ["Estimate Sample Size", "Estimate MDE"], horizontal=True, help="Choose whether to solve for sample size or for the minimum detectable effect.")
    
    p_A = st.number_input("Baseline rate (p_A)", 0.0001, 0.999, 0.05, 0.001, format="%.4f", help="The conversion rate of your control group (e.g., 0.05 for 5%).")
    
    if mode == "Estimate Sample Size":
        uplift = st.number_input("Expected uplift", 0.0001, 0.999, 0.10, 0.01, format="%.4f", help="The relative improvement you expect to see (e.g., 0.10 for a 10% lift).")
        p_B_check = p_A * (1 + uplift)
        if p_B_check > 1.0:
            st.warning(f"Variant rate ({p_B_check:.2%}) > 100%. Invalid input.")
    else: # MDE
        fixed_n = st.number_input("Fixed sample size per variant", 100, value=10000, step=100, help="The number of users you have available for each group.")

    if methodology == "Bayesian":
        st.subheader("Bayesian Settings")
        thresh = st.slider("Posterior threshold", 0.8, 0.99, 0.95, help="The probability (P(B > A)) required to declare a winner.")
        desired_power = st.slider("Desired Power", 0.5, 0.99, 0.8, help="The chance of detecting the uplift if it's real. 80% is common.")
        sims = st.slider("Simulations", 100, 2000, 500, help="Number of simulated A/B tests to run. More is more accurate but slower.")
        samples = st.slider("Posterior samples", 500, 3000, 1000, help="Samples drawn from the posterior distribution in each simulation.")
        use_hist = st.checkbox("Auto-priors from history", help="Use historical data to create an informative prior.")
        if use_hist:
            hist_conv = st.number_input("Hist. successes", 0, value=50)
            hist_n = st.number_input("Hist. users", 1, value=1000)
            a0, b0 = hist_conv, hist_n - hist_conv
        else:
            a0 = st.number_input("Alpha prior", 0.0, value=1.0, help="Prior belief in successes (default is 1).")
            b0 = st.number_input("Beta prior", 0.0, value=1.0, help="Prior belief in failures (default is 1).")
    else: # Frequentist
        st.subheader("Frequentist Settings")
        alpha = st.slider("Significance α", 0.01, 0.10, 0.05, help="Your tolerance for a false positive. 0.05 is standard for 95% confidence.")
        desired_power = st.slider("Desired Power (1-β)", 0.5, 0.99, 0.8, help="The chance of detecting the uplift if it's real. 80% is common.")

    st.header("2. Duration & Geo Spend")
    weekly_traffic = st.number_input("Weekly traffic", 1, 1000000, 20000, help="Total users entering the experiment each week (before the 50/50 split).")
    calculate_geo_spend = st.checkbox("Calculate Geo Spend", value=False, help="Enable to plan ad spend for a geo-based test.")
    
    if calculate_geo_spend:
        with st.expander("Configure Geo Spend Data", expanded=True):
            # NEW: Multiselect to choose which regions to include in the test
            selected_regions = st.multiselect(
                "Select regions to include in the test",
                options=GEO_DEFAULTS["Region"].tolist(),
                default=GEO_DEFAULTS["Region"].tolist(),
                help="Deselect regions to exclude them from the test. Weights will be re-normalized."
            )
            
            spend_mode = st.radio("Weighting Mode", ["Population-based", "Equal", "Custom"], index=0, horizontal=True, help="Choose how to distribute the sample size across the *selected* regions.")
            
            if spend_mode == "Custom":
                st.caption("Edit weights and CPMs for selected regions. Weights must sum to 100%.")
                
                # Filter the default data for selected regions to create a base for the editor
                editor_base_df = GEO_DEFAULTS[GEO_DEFAULTS["Region"].isin(selected_regions)].copy()
                
                # Initialize or update session state for the custom data editor
                if 'geo_df_custom' not in st.session_state:
                    st.session_state.geo_df_custom = editor_base_df
                else:
                    # Keep existing custom values for regions that are still selected
                    st.session_state.geo_df_custom = st.session_state.geo_df_custom[st.session_state.geo_df_custom["Region"].isin(selected_regions)]

                edited_df = st.data_editor(st.session_state.geo_df_custom, num_rows="dynamic")
                st.session_state.geo_df_custom = edited_df

    submit = st.form_submit_button("Run Calculation")

# --- Main Application Logic ---
if submit:
    st.header("Results")
    req_n = None
    
    # ... (Calculation logic for req_n is unchanged)
    if methodology == "Frequentist":
        if mode == "Estimate Sample Size":
            req_n = calculate_sample_size_frequentist(p_A, uplift, desired_power, alpha)
        else:
            req_n = fixed_n
    else: # Bayesian
        if mode == "Estimate Sample Size":
            results = simulate_power(p_A, uplift, thresh, desired_power, sims, samples, a0, b0)
            if results and results[-1][1] >= desired_power:
                req_n = results[-1][0]
        else:
            req_n = fixed_n

    # Display main results first
    if mode == "Estimate Sample Size":
        st.subheader("📈 Required Sample Size")
        if req_n:
            st.success(f"**{req_n:,} per variant**")
        else:
            st.error("Unable to compute sample size. Try increasing uplift/power or check inputs.")
    else: # MDE Mode
        st.subheader("📉 Minimum Detectable Effect (MDE)")
        if methodology == "Frequentist":
            mde_results = calculate_mde_frequentist(p_A, fixed_n, desired_power, alpha)
        else:
            mde_results = simulate_mde(p_A, thresh, desired_power, sims, samples, a0, b0, fixed_n)
        
        if mde_results and mde_results[-1][1] >= desired_power:
            mde, achieved_power = mde_results[-1]
            st.success(f"**{mde:.2%}** relative uplift (achieved {achieved_power:.1%} power)")
        else:
            st.warning("Could not reach desired power with the given sample size.")

    # --- MODIFIED GEO SPEND LOGIC ---
    if calculate_geo_spend and req_n:
        st.subheader("💰 Geo Ad Spend")
        
        if not selected_regions:
            st.error("Please select at least one region to calculate geo spend.")
        else:
            # 1. Filter the master data based on user's selection
            active_geo_data = GEO_DEFAULTS[GEO_DEFAULTS["Region"].isin(selected_regions)].copy()
            
            if spend_mode == "Population-based":
                # 2. Re-normalize weights of the remaining regions
                active_geo_data["Weight"] /= active_geo_data["Weight"].sum()
                geo_df = active_geo_data
            elif spend_mode == "Equal":
                active_geo_data["Weight"] = 1 / len(active_geo_data)
                geo_df = active_geo_data
            else: # Custom
                geo_df = st.session_state.geo_df_custom
                if not np.isclose(geo_df["Weight"].sum(), 1.0):
                    st.error("Custom weights must sum to 1.0. Please adjust in the editor.")
                    geo_df = pd.DataFrame() # Invalidate df

            if not geo_df.empty:
                total_users = req_n * 2
                geo_df["Users"] = (geo_df["Weight"] * total_users).astype(int)
                geo_df["Impressions (k)"] = geo_df["Users"] / 1000
                geo_df["Spend (£)"] = geo_df["Impressions (k)"] * geo_df["CPM (£)"]
                
                style = {"Weight": "{:.1%}", "Users": "{:,.0f}", "CPM (£)": "£{:.2f}", "Impressions (k)": "{:,.1f}", "Spend (£)": "£{:,.2f}"}
                st.dataframe(geo_df.style.format(style))
                st.download_button("Download CSV", geo_df.to_csv(index=False), file_name="geo_spend_plan.csv")
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(geo_df["Region"], geo_df["Spend (£)"])
                ax.set_xlabel("Spend (£)")
                ax.set_title("Geo Spend Breakdown (Selected Regions)")
                plt.tight_layout()
                st.pyplot(fig)

    if req_n:
        st.subheader("🗓️ Estimated Test Duration")
        if weekly_traffic > 0:
            weeks = (req_n * 2) / weekly_traffic
            st.info(f"You will need approximately **{weeks:.1f} weeks** to reach the required total sample size.")
        else:
            st.warning("Weekly traffic must be > 0 to estimate duration.")

    # --- Other plots remain unchanged ---
    if methodology == "Frequentist" and mode == "Estimate Sample Size" and req_n:
        st.subheader("🔬 Sample Size vs. Uplift")
        # ... (plotting logic is fine)

else:
    st.info("Adjust the parameters in the sidebar and click 'Run Calculation'.")

# --- Explanations Section (Unchanged) ---
st.markdown("---")
with st.expander("ℹ️ About the Methodologies & Geo Testing"):
    st.markdown("""
    #### Bayesian vs. Frequentist Approaches
    - **Bayesian (Simulation-Based):** A modern approach that answers: *"What is the probability that my variant is better?"*
    - **Frequentist (Formula-Based):** The traditional method using p-values and significance levels ($α$).
    
    ---
    #### About Geo Testing Ad Spend
    Geo testing is for channels where you can't randomize individual users (e.g., TV, radio). Instead, you randomize by geographic region.
    - **How it works:** This calculator takes the total required sample size and distributes it across the **regions you select**. You can switch regions on or off using the multiselect box.
    - **Weighting:** For the selected regions, you can use an equal split, a split based on their relative population, or a fully custom split. The calculator automatically handles re-calculating the weights.
    - **The Output:** The final table and chart show the estimated ad spend required in each *active* region to run a properly powered test.
    """)
