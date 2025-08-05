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
    n_A, n_B = n, n
    rng = np.random.default_rng(seed=42)
    conversions_A = rng.binomial(n_A, p_A, size=simulations)
    conversions_B = rng.binomial(n_B, p_B, size=simulations)
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
    effect_size_norm = abs(p_B - p_A) / se
    z_alpha = norm.ppf(1 - alpha / 2)
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
if 'show_region_dialog' not in st.session_state:
    st.session_state.show_region_dialog = False
if 'geo_df_custom' not in st.session_state:
    st.session_state.geo_df_custom = GEO_DEFAULTS.copy()

# --- UI ---
st.title("⚙️ Pre-Test Power Calculator")

# --- Sidebar Controls ---
with st.sidebar.form("params_form"):
    st.header("1. Main Parameters")
    methodology = st.radio("Methodology", ["Bayesian", "Frequentist"], horizontal=True)
    mode = st.radio("Planning Mode", ["Estimate Sample Size", "Estimate MDE"], horizontal=True)
    p_A = st.number_input("Baseline rate (p_A)", 0.0001, 0.999, 0.05, 0.001, format="%.4f")
    
    if mode == "Estimate Sample Size":
        uplift = st.number_input("Expected uplift", 0.0001, 0.999, 0.10, 0.01, format="%.4f")
    else: # MDE
        fixed_n = st.number_input("Fixed sample size per variant", 100, value=10000, step=100)

    if methodology == "Bayesian":
        st.subheader("Bayesian Settings")
        thresh, desired_power = st.slider("Posterior threshold", 0.8, 0.99, 0.95), st.slider("Desired Power", 0.5, 0.99, 0.8)
        sims, samples = st.slider("Simulations", 100, 2000, 500), st.slider("Posterior samples", 500, 3000, 1000)
    else: # Frequentist
        st.subheader("Frequentist Settings")
        alpha, desired_power = st.slider("Significance α", 0.01, 0.10, 0.05), st.slider("Desired Power (1-β)", 0.5, 0.99, 0.8)
    
    st.header("2. Duration")
    weekly_traffic = st.number_input("Weekly traffic", 1, 1000000, 20000)

    submit = st.form_submit_button("Run Calculation", type="primary")

st.sidebar.header("3. Geo Spend Configuration")
calculate_geo_spend = st.sidebar.checkbox("Calculate Geo Spend", value=True)
if calculate_geo_spend:
    # --- NEW: Region selection via dialog ---
    if st.sidebar.button("Change Active Regions..."):
        st.session_state.show_region_dialog = True
    
    with st.sidebar.expander("Currently Active Regions", expanded=True):
        if st.session_state.selected_regions:
             # Create a two-column layout for the list
            col1, col2 = st.columns(2)
            midpoint = len(st.session_state.selected_regions) // 2 + len(st.session_state.selected_regions) % 2
            with col1:
                for region in st.session_state.selected_regions[:midpoint]:
                    st.caption(f"• {region}")
            with col2:
                for region in st.session_state.selected_regions[midpoint:]:
                    st.caption(f"• {region}")
        else:
            st.caption("None selected")

    spend_mode = st.sidebar.radio("Weighting Mode", ["Population-based", "Equal", "Custom"], index=0, horizontal=True)

# --- NEW: Region Selection Dialog Logic ---
if st.session_state.get("show_region_dialog", False):
    with st.dialog("Select Active Regions", width="large"):
        st.write("Choose the regions to include in your geo-test.")
        
        # Helper buttons
        c1, c2, _, c3 = st.columns([1, 1, 3, 1])
        if c1.button("Select All", use_container_width=True):
            st.session_state.selected_regions = ALL_REGIONS
            st.rerun()
        if c2.button("Deselect All", use_container_width=True):
            st.session_state.selected_regions = []
            st.rerun()

        # Checkbox for each region, in three columns
        cols = st.columns(3)
        temp_selections = []
        for i, region in enumerate(ALL_REGIONS):
            with cols[i % 3]:
                if st.checkbox(region, value=(region in st.session_state.selected_regions), key=f"check_{region}"):
                    temp_selections.append(region)
        
        st.session_state.selected_regions = temp_selections
        
        if c3.button("Confirm", type="primary", use_container_width=True):
            st.session_state.show_region_dialog = False
            st.rerun()

# --- Interactive Custom Editor in Main Panel ---
if calculate_geo_spend and spend_mode == 'Custom':
    st.subheader("🛠️ Edit Custom Weights & CPMs")
    st.caption("Adjust values for the regions selected in the sidebar. The editor is live and saves automatically.")
    
    editor_display_df = st.session_state.geo_df_custom[st.session_state.geo_df_custom['Region'].isin(st.session_state.selected_regions)].copy()
    
    if not editor_display_df.empty:
        edited_df = st.data_editor(editor_display_df, num_rows="dynamic", use_container_width=True, key="custom_geo_editor")
        current_sum = edited_df['Weight'].sum()
        delta = current_sum - 1.0
        st.metric(label="Current Weight Sum", value=f"{current_sum:.2%}", delta=f"{delta:.2%} from target")
        if not np.isclose(current_sum, 1.0):
            st.warning("Sum of weights must be 100%.")
        st.session_state.geo_df_custom.update(edited_df)
    else:
        st.warning("Please select at least one region to configure custom weights.")
st.markdown("---")

# --- Main Application Logic ---
if submit:
    st.header("Results")
    req_n = None
    
    if mode == "Estimate Sample Size":
        st.subheader("📈 Required Sample Size")
        if methodology == "Frequentist":
            req_n = calculate_sample_size_frequentist(p_A, uplift, desired_power, alpha)
        else:
            b_results = simulate_power(p_A, uplift, thresh, desired_power, sims, samples, 1, 1)
            if b_results and b_results[-1][1] >= desired_power:
                req_n = b_results[-1][0]
        if req_n: st.success(f"**{req_n:,} per variant**")
        else: st.error("Unable to compute sample size.")
    else:
        req_n = fixed_n
        st.subheader("📉 Minimum Detectable Effect (MDE)")
        if methodology == "Frequentist":
            mde_results = calculate_mde_frequentist(p_A, fixed_n, desired_power, alpha)
        else:
            mde_results = simulate_mde(p_A, thresh, desired_power, sims, samples, 1, 1, fixed_n)
        if mde_results and mde_results[-1][1] >= desired_power:
            mde, achieved_power = mde_results[-1]
            st.success(f"**{mde:.2%}** relative uplift (achieved {achieved_power:.1%} power)")
        else:
            st.warning("Could not reach desired power with the given sample size.")

    if calculate_geo_spend and req_n:
        st.subheader("💰 Geo Ad Spend")
        if not st.session_state.selected_regions:
            st.error("Please select at least one region to calculate geo spend.")
        else:
            if spend_mode == "Custom":
                geo_df = st.session_state.geo_df_custom[st.session_state.geo_df_custom['Region'].isin(st.session_state.selected_regions)].copy()
                if not np.isclose(geo_df['Weight'].sum(), 1.0):
                    st.error("Final check failed: Custom weights must sum to 1.0.")
                    geo_df = pd.DataFrame()
            else:
                base_df = GEO_DEFAULTS[GEO_DEFAULTS['Region'].isin(st.session_state.selected_regions)].copy()
                if spend_mode == "Population-based": base_df["Weight"] /= base_df["Weight"].sum()
                else: base_df["Weight"] = 1 / len(base_df)
                geo_df = base_df
            if not geo_df.empty:
                total_users = req_n * 2
                geo_df["Users"] = (geo_df["Weight"] * total_users).astype(int)
                geo_df["Impressions (k)"] = geo_df["Users"] / 1000
                geo_df["Spend (£)"] = geo_df["Impressions (k)"] * geo_df["CPM (£)"]
                style = {"Weight": "{:.1%}", "Users": "{:,.0f}", "CPM (£)": "£{:.2f}", "Impressions (k)": "{:,.1f}", "Spend (£)": "£{:,.2f}"}
                st.dataframe(geo_df.style.format(style), use_container_width=True)
                st.download_button("Download CSV", geo_df.to_csv(index=False), file_name="geo_spend_plan.csv")
                fig, ax = plt.subplots()
                ax.barh(geo_df["Region"], geo_df["Spend (£)"])
                ax.set_xlabel("Spend (£)"); ax.set_title("Geo Spend Breakdown (Selected Regions)")
                plt.tight_layout(); st.pyplot(fig)

    if req_n:
        st.subheader("🗓️ Estimated Test Duration")
        if weekly_traffic > 0:
            weeks = (req_n * 2) / weekly_traffic
            st.info(f"You will need approximately **{weeks:.1f} weeks** to reach the required total sample size.")
else:
    st.info("Set your parameters in the sidebar and click 'Run Calculation'. If using geo-spend, configure regions and weights.")
