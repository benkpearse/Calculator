import streamlit as st
import numpy as np
from scipy.stats import beta, norm
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple

# 1. Set Page Configuration
st.set_page_config(
    page_title="A/B Test Power Calculator",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Core Calculation Functions (Unchanged) ---
@st.cache_data
def run_simulation(n: int, p_A: float, p_B: float, simulations: int, samples: int, alpha_prior: float, beta_prior: float, thresh: float) -> float:
    n_A, n_B = n, n; rng = np.random.default_rng(seed=42)
    conversions_A = rng.binomial(n_A, p_A, size=simulations); conversions_B = rng.binomial(n_B, p_B, size=simulations)
    alpha_post_A, beta_post_A = alpha_prior + conversions_A, n_A - conversions_A + beta_prior
    alpha_post_B, beta_post_B = alpha_prior + conversions_B, n_B - conversions_B + beta_prior
    post_samples_A = beta.rvs(alpha_post_A, beta_post_A, size=(samples, simulations), random_state=rng)
    post_samples_B = beta.rvs(alpha_post_B, beta_post_B, size=(samples, simulations), random_state=rng)
    prob_B_better = np.mean(post_samples_B > post_samples_A, axis=0)
    return np.mean(prob_B_better > thresh)

@st.cache_data
def simulate_power(p_A: float, uplift: float, thresh: float, desired_power: float, simulations: int, samples: int, alpha_prior: float, beta_prior: float) -> List[Tuple[int, float]]:
    p_B = p_A * (1 + uplift)
    if p_B > 1.0: return []
    results, n, power, MAX_SAMPLE_SIZE = [], 100, 0, 5_000_000
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
    results = []
    with st.spinner("Running simulations for MDE..."):
        for uplift in np.linspace(0.01, 0.50, 20):
            p_B = p_A * (1 + uplift)
            if p_B > 1.0: continue
            power = run_simulation(fixed_n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh)
            results.append((uplift, power))
            if power >= desired_power: break
    return results

@st.cache_data
def calculate_power_frequentist(p_A: float, p_B: float, n: int, alpha: float = 0.05) -> float:
    if p_B < 0 or p_B > 1.0: return 0.0
    se = np.sqrt(p_A * (1 - p_A) / n + p_B * (1 - p_B) / n)
    if se == 0: return 1.0
    effect_size_norm = abs(p_B - p_A) / se; z_alpha = norm.ppf(1 - alpha / 2)
    return norm.cdf(effect_size_norm - z_alpha) + norm.cdf(-effect_size_norm - z_alpha)

@st.cache_data
def calculate_sample_size_frequentist(p_A: float, uplift: float, power_target: float = 0.8, alpha: float = 0.05) -> int | None:
    p_B = p_A * (1 + uplift)
    if p_B >= 1: return None
    n, power, MAX_SAMPLE_SIZE = 100, 0, 5_000_000
    while power < power_target and n < MAX_SAMPLE_SIZE:
        power = calculate_power_frequentist(p_A, p_B, n, alpha)
        if power >= power_target: return int(n)
        if n < 1000: n += 50
        elif n < 20000: n = int(n * 1.2)
        else: n = int(n * 1.1)
    if n >= MAX_SAMPLE_SIZE: return None
    return int(n)

@st.cache_data
def calculate_mde_frequentist(p_A: float, n: int, power_target: float = 0.8, alpha: float = 0.05) -> List[Tuple[float, float]]:
    results = []
    for uplift in np.linspace(0.001, 0.50, 100):
        p_B = p_A * (1 + uplift)
        if p_B > 1.0: continue
        power = calculate_power_frequentist(p_A, p_B, n, alpha)
        results.append((uplift, power))
        if power >= power_target: break
    return results

# --- Geo Testing Data ---
GEO_DEFAULTS = pd.DataFrame({
    "Region": ["North East", "North West", "Yorkshire and the Humber", "East Midlands", "West Midlands", "East of England", "London", "South East", "South West", "Wales", "Scotland", "Northern Ireland"],
    "Weight": [0.03, 0.09, 0.07, 0.07, 0.09, 0.10, 0.18, 0.16, 0.07, 0.04, 0.07, 0.03],
    "CPM (£)": [7.50, 8.00, 8.25, 7.00, 7.80, 8.10, 12.00, 10.00, 7.60, 6.90, 9.00, 8.50]
})
ALL_REGIONS = GEO_DEFAULTS["Region"].tolist()

# --- Initialize Session State ---
if 'selected_regions' not in st.session_state:
    st.session_state.selected_regions = ALL_REGIONS
if 'geo_df_custom' not in st.session_state:
    st.session_state.geo_df_custom = GEO_DEFAULTS.copy()

# --- UI ---
st.title("⚙️ Pre-Test Power Calculator")

with st.expander("What is Power Analysis? Click here to learn more.", expanded=False):
    st.markdown("""
    Power analysis is a statistical method used **before** an A/B test to estimate the resources needed. It helps you design a test that is both effective and efficient.
    - **Why is it important?** Without proper planning, you might run a test that is too short to detect a real improvement (a "false negative"), or a test that is unnecessarily long, wasting time and resources.
    #### Key Concepts
    - **Sample Size:** The number of users or sessions required in each group (e.g., 'Control' and 'Variant').
    - **Statistical Power (or Sensitivity):** The probability of detecting a real effect, if one truly exists. A power of 80% means you have an 80% chance of detecting a genuine uplift.
    - **Minimum Detectable Effect (MDE):** The smallest improvement you want your test to be able to detect.
    #### How to Use This Tool
    1.  **Set Inputs:** Use the sidebar to enter your test parameters.
    2.  **Configure Geo-Test (Optional):** Use the main panel to select active regions and set custom weights or costs.
    3.  **Calculate:** Click "Run Calculation" to see the summary of required resources.
    """)

with st.sidebar.form("params_form"):
    st.header("1. Main Parameters")
    methodology = st.radio("Methodology", ["Bayesian", "Frequentist"], horizontal=True, help="Choose the statistical approach.")
    mode = st.radio("Planning Mode", ["Estimate Sample Size", "Estimate MDE"], horizontal=True, help="Solve for sample size or minimum detectable effect.")
    p_A = st.number_input("Baseline rate (p_A)", 0.0001, 0.999, 0.05, 0.001, format="%.4f", help="Conversion rate of the control group (e.g., 0.05 for 5%).")
    if mode == "Estimate Sample Size":
        uplift = st.number_input("Expected uplift", 0.0001, 0.999, 0.10, 0.01, format="%.4f", help="Relative improvement you expect (e.g., 0.10 for a 10% lift).")
    else:
        fixed_n = st.number_input("Fixed sample size per variant", 100, value=10000, step=100, help="Users available for each group.")
    if methodology == "Bayesian":
        st.subheader("Bayesian Settings")
        thresh, desired_power = st.slider("Posterior threshold", 0.8, 0.99, 0.95, help="P(B > A) required to declare a winner."), st.slider("Desired Power", 0.5, 0.99, 0.8, help="Chance of detecting the uplift if it's real.")
        sims, samples = st.slider("Simulations", 100, 2000, 500, help="Number of simulated A/B tests."), st.slider("Posterior samples", 500, 3000, 1000, help="Samples from the posterior distribution.")
    else:
        st.subheader("Frequentist Settings")
        alpha, desired_power = st.slider("Significance α", 0.01, 0.10, 0.05, help="Tolerance for a false positive."), st.slider("Desired Power (1-β)", 0.5, 0.99, 0.8, help="Chance of detecting the uplift if it's real.")
    
    st.header("2. Optional Calculations")
    estimate_duration = st.checkbox("Estimate Test Duration", value=True)
    if estimate_duration:
        weekly_traffic = st.number_input("Weekly traffic", min_value=1, value=20000, help="Total users entering the experiment each week (before 50/50 split).")
    else:
        weekly_traffic = 0

    submit = st.form_submit_button("Run Calculation", type="primary")

st.sidebar.header("Geo Spend Configuration")
calculate_geo_spend = st.sidebar.checkbox("Calculate Geo Spend", value=True, help="Enable to plan ad spend for a geo-based test.")
if calculate_geo_spend:
    spend_mode = st.sidebar.radio("Weighting Mode", ["Population-based", "Equal", "Custom"], index=0, horizontal=True, help="How to distribute sample size across active regions.")

if calculate_geo_spend:
    with st.expander("Configure Active Regions and Custom Data", expanded=False):
        st.write("First, select regions, then click 'Confirm'. For 'Custom' mode, the editor will then appear.")
        with st.form("region_selection_form"):
            temp_selections = []
            cols = st.columns(3)
            for i, region in enumerate(ALL_REGIONS):
                with cols[i % 3]:
                    if st.checkbox(region, value=(region in st.session_state.selected_regions), key=f"check_{region}"):
                        temp_selections.append(region)
            
            submitted = st.form_submit_button("Confirm Region Selection")
            if submitted:
                st.session_state.selected_regions = temp_selections
                current_custom_regions = st.session_state.geo_df_custom['Region'].tolist()
                for region in st.session_state.selected_regions:
                    if region not in current_custom_regions:
                        new_row = GEO_DEFAULTS[GEO_DEFAULTS['Region'] == region]
                        st.session_state.geo_df_custom = pd.concat([st.session_state.geo_df_custom, new_row], ignore_index=True)
                st.rerun()

        if spend_mode == 'Custom':
            st.markdown("---")
            
            # --- DEFINITIVE FIX FOR THE CUSTOM EDITOR ---
            # 1. First, if the editor's state exists, convert it to a DataFrame.
            if "custom_geo_editor" in st.session_state:
                edited_data_from_state = pd.DataFrame(st.session_state["custom_geo_editor"])
                
                # 2. Use 'Region' as the key to robustly update the master DataFrame.
                # This prevents index ambiguity and solves the ValueError.
                master_df = st.session_state.geo_df_custom.set_index('Region')
                updates_df = edited_data_from_state.set_index('Region')
                master_df.update(updates_df)
                st.session_state.geo_df_custom = master_df.reset_index()

            # 3. Create the dataframe to be displayed from the (now updated) master dataframe.
            editor_display_df = st.session_state.geo_df_custom[st.session_state.geo_df_custom['Region'].isin(st.session_state.selected_regions)].copy()
            
            if not editor_display_df.empty:
                # 4. Finally, render the editor. Its state is now managed correctly.
                st.data_editor(
                    editor_display_df, 
                    num_rows="dynamic", 
                    use_container_width=True, 
                    key="custom_geo_editor"
                )
                
                current_sum = editor_display_df['Weight'].sum()
                st.metric(label="Current Weight Sum", value=f"{current_sum:.2%}", delta=f"{(current_sum - 1.0):.2%} from target")
                if not np.isclose(current_sum, 1.0):
                    st.warning("Sum of weights must be 100%.")
            else:
                st.warning("Please select at least one region and click 'Confirm' to configure custom weights.")

st.markdown("---")

if submit:
    st.header("Results")
    req_n, total_spend, weeks = None, None, None
    
    if mode == "Estimate Sample Size":
        if methodology == "Frequentist": req_n = calculate_sample_size_frequentist(p_A, uplift, desired_power, alpha)
        else:
            b_results = simulate_power(p_A, uplift, thresh, desired_power, sims, samples, 1, 1)
            if b_results and b_results[-1][1] >= desired_power: req_n = b_results[-1][0]
    else: req_n = fixed_n
    
    if calculate_geo_spend and req_n:
        if st.session_state.selected_regions:
            geo_df = pd.DataFrame()
            if spend_mode == "Custom":
                # For custom mode, the source of truth is our master custom dataframe
                geo_df = st.session_state.geo_df_custom[st.session_state.geo_df_custom['Region'].isin(st.session_state.selected_regions)].copy()
                if not np.isclose(geo_df['Weight'].sum(), 1.0):
                    st.error("Final check failed: Custom weights must sum to 1.0."); geo_df = pd.DataFrame()
            else:
                base_df = GEO_DEFAULTS[GEO_DEFAULTS['Region'].isin(st.session_state.selected_regions)].copy()
                if not base_df.empty:
                    if spend_mode == "Population-based": base_df["Weight"] /= base_df["Weight"].sum()
                    else: base_df["Weight"] = 1 / len(base_df)
                geo_df = base_df
            
            if not geo_df.empty:
                geo_df["Users"] = (geo_df["Weight"] * (req_n * 2)).astype(int)
                geo_df["Impressions (k)"] = geo_df["Users"] / 1000
                geo_df["Spend (£)"] = geo_df["Impressions (k)"] * geo_df["CPM (£)"]
                total_spend = geo_df['Spend (£)'].sum()

    if weekly_traffic and weekly_traffic > 0 and req_n:
        weeks = (req_n * 2) / weekly_traffic

    if req_n:
        with st.container(border=True):
            st.subheader("Executive Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sample Size (per Variant)", f"{req_n:,}")
            col2.metric("Total Users Required", f"{(req_n * 2):,}")
            if total_spend is not None: col3.metric("Total Estimated Ad Spend", f"£{total_spend:,.0f}")
            else: col3.metric("Total Estimated Ad Spend", "N/A")
            
            st.markdown("---")
            summary_text = f"To confidently detect the specified effect, this test requires **{req_n*2:,} total users**."
            if total_spend is not None: summary_text += f" This corresponds to an estimated ad spend of **£{total_spend:,.0f}**."
            if weeks is not None: summary_text += f" At the specified traffic rate, the test will take approximately **{weeks:.1f} weeks**."
            st.info(summary_text)
    else:
        st.error("Could not determine the required sample size with the provided inputs.")

    if mode == "Estimate MDE":
        st.subheader("📉 Minimum Detectable Effect")
        if methodology == "Frequentist": mde_results = calculate_mde_frequentist(p_A, fixed_n, desired_power, alpha)
        else: mde_results = simulate_mde(p_A, thresh, desired_power, sims, samples, 1, 1, fixed_n)
        if mde_results and mde_results[-1][1] >= desired_power:
            mde, achieved_power = mde_results[-1]
            st.success(f"With **{fixed_n:,} users** per variant, the smallest uplift you can reliably detect is **{mde:.2%}** (with {achieved_power:.1%} power).")
        else:
            st.warning("Could not reach desired power with the given sample size.")
    
    if calculate_geo_spend and total_spend is not None:
        with st.expander("View Geo Spend Breakdown"):
            st.write("**Spend Breakdown by Region**")
            style = {"Weight": "{:.1%}", "Users": "{:,.0f}", "CPM (£)": "£{:.2f}", "Impressions (k)": "{:,.1f}", "Spend (£)": "£{:,.2f}"}
            st.dataframe(geo_df.style.format(style), use_container_width=True)
            st.download_button("Download CSV", geo_df.to_csv(index=False), file_name="geo_spend_plan.csv")
            fig, ax = plt.subplots(); ax.barh(geo_df["Region"], geo_df["Spend (£)"])
            ax.set_xlabel("Spend (£)"); ax.set_title("Geo Spend Breakdown"); plt.tight_layout(); st.pyplot(fig)

    if methodology == "Frequentist" and mode == "Estimate Sample Size" and req_n:
        with st.expander("View Sample Size vs. Uplift Sensitivity"):
            uplifts_plot = np.linspace(uplift * 0.5, uplift * 2.0, 50)
            sizes = [calculate_sample_size_frequentist(p_A, u, desired_power, alpha) for u in uplifts_plot if u > 0 and p_A*(1+u) <= 1]
            valid_uplifts = [u for u in uplifts_plot if u > 0 and p_A*(1+u) <= 1 and calculate_sample_size_frequentist(p_A, u, desired_power, alpha) is not None]
            sizes = [s for s in sizes if s is not None]
            if valid_uplifts:
                fig2, ax2 = plt.subplots()
                ax2.plot([u * 100 for u in valid_uplifts], sizes); ax2.axvline(uplift * 100, linestyle='--', color='red', label=f"Your Target ({uplift:.1%})")
                ax2.set_xlabel("Uplift (%)"); ax2.set_ylabel("Sample Size per Variant")
                ax2.set_title("Sample Size vs Uplift"); ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6); st.pyplot(fig2)

else:
    st.info("Set your parameters in the sidebar and click 'Run Calculation'.")
