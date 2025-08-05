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

# --- Core Calculation Functions (Unchanged) ---
@st.cache_data
def run_simulation(n: int, p_A: float, p_B: float, simulations: int, samples: int, alpha_prior: float, beta_prior: float, num_variants: int) -> float:
    rng = np.random.default_rng(seed=42)
    true_rates = [p_A] + [p_B] + [p_A] * (num_variants - 1)
    post_samples_all_groups = []
    for rate in true_rates:
        conversions = rng.binomial(n, rate, size=simulations)
        alpha_post, beta_post = alpha_prior + conversions, beta_prior + n - conversions
        post_samples_all_groups.append(beta.rvs(alpha_post, beta_post, size=(samples, simulations), random_state=rng))
    stacked_samples = np.stack(post_samples_all_groups)
    best_variant_indices = np.argmax(stacked_samples, axis=0)
    return np.mean(best_variant_indices == 1)

@st.cache_data
def simulate_power(p_A: float, uplift: float, desired_power: float, simulations: int, samples: int, alpha_prior: float, beta_prior: float, num_variants: int) -> List[Tuple[int, float]]:
    p_B = p_A * (1 + uplift)
    if p_B > 1.0: return []
    results, n, power, MAX_SAMPLE_SIZE = [], 100, 0, 5_000_000
    n_lower, n_upper = 0, 0
    with st.spinner("Stage 1/2: Finding approximate sample size range..."):
        while power < desired_power and n < MAX_SAMPLE_SIZE:
            power = run_simulation(n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, num_variants)
            results.append((n, power))
            if power >= desired_power:
                n_upper, n_lower = n, (results[-2][0] if len(results) > 1 else n // 2)
                break
            if n < 1000: n += 100
            elif n < 20000: n = int(n * 1.5)
            else: n = int(n * 1.25)
    if n_upper == 0: return results
    with st.spinner(f"Stage 2/2: Refining sample size between {n_lower:,} and {n_upper:,}..."):
        for _ in range(5):
            n_mid = (n_lower + n_upper) // 2
            if n_mid == n_lower: break
            power_mid = run_simulation(n_mid, p_A, p_B, simulations, samples, alpha_prior, beta_prior, num_variants)
            results.append((n_mid, power_mid))
            if power_mid >= desired_power: n_upper = n_mid
            else: n_lower = n_mid
    final_results = sorted([res for res in results if res[1] >= desired_power], key=lambda x: x[0])
    return final_results if final_results else results

@st.cache_data
def simulate_mde_bayesian(p_A: float, fixed_n: int, desired_power: float, simulations: int, samples: int, alpha_prior: float, beta_prior: float, num_variants: int) -> List[Tuple[float, float]]:
    results = []
    with st.spinner("Running Bayesian MDE simulations..."):
        for uplift in np.linspace(0.01, 0.50, 25):
            p_B = p_A * (1 + uplift)
            if p_B > 1.0: continue
            power = run_simulation(fixed_n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, num_variants)
            results.append((uplift, power))
            if power >= desired_power: break
    return results

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

def reset_app_state():
    """Clears all session state variables to reset the app."""
    st.session_state.clear()

# --- UI ---
st.title("⚙️ A/B/n Pre-Test Power Calculator")

with st.expander("What is Power Analysis? Click here to learn more.", expanded=False):
    st.markdown("""...""") # Content unchanged

st.sidebar.button("Reset All Settings", on_click=reset_app_state, use_container_width=True)
st.sidebar.markdown("---")

st.sidebar.header("1. Main Parameters")
num_variants = st.sidebar.number_input("Number of Variants (excluding control)", min_value=1, max_value=10, value=1, key='num_variants', help="An A/B test has 1 variant. An A/B/C test has 2 variants.")
methodology = st.sidebar.radio("Methodology", ["Bayesian", "Frequentist"], horizontal=True, key='methodology', help="Choose the statistical approach.")
mode = st.sidebar.radio("Planning Mode", ["Estimate Sample Size", "Estimate MDE"], horizontal=True, key='mode', help="Solve for sample size or minimum detectable effect.")
p_A = st.sidebar.number_input("Baseline rate (p_A)", 0.0001, 0.999, 0.05, 0.001, format="%.4f", key='p_A', help="Conversion rate of the control group.")
if mode == "Estimate Sample Size":
    uplift = st.sidebar.number_input("Expected uplift", 0.0001, 0.999, 0.10, 0.01, format="%.4f", key='uplift', help="Relative improvement you want to detect in the winning variant.")
else:
    fixed_n = st.sidebar.number_input("Fixed sample size per group", 100, value=10000, step=100, key='fixed_n', help="Users available for the control and EACH variant.")
if methodology == "Bayesian":
    st.sidebar.subheader("Bayesian Settings")
    desired_power = st.sidebar.slider("Desired Power", 0.5, 0.99, 0.8, key='desired_power_b', help="The probability of correctly identifying the best performing variant.")
    sims = st.sidebar.slider("Simulations", 100, 2000, 500, key='sims')
    samples = st.sidebar.slider("Posterior samples", 500, 3000, 1000, key='samples')
else:
    st.sidebar.subheader("Frequentist Settings")
    alpha = st.sidebar.slider("Significance α (Family-wise)", 0.01, 0.10, 0.05, key='alpha', help="Overall chance of a false positive. Auto-adjusted for multiple comparisons.")
    desired_power = st.sidebar.slider("Desired Power (1-β)", 0.5, 0.99, 0.8, key='desired_power_f')

st.sidebar.header("2. Optional Calculations")
estimate_duration = st.sidebar.checkbox("Estimate Test Duration", value=True, key='estimate_duration')
if estimate_duration:
    weekly_traffic = st.sidebar.number_input("Total weekly traffic for test", min_value=1, value=20000, key='weekly_traffic', help="All users entering the experiment, to be split across all groups.")
else:
    weekly_traffic = 0

st.sidebar.header("3. Geo Spend Configuration")
calculate_geo_spend = st.sidebar.checkbox("Calculate Geo Spend", value=True, key='calculate_geo_spend', help="Enable to plan ad spend for a geo-based test.")
if calculate_geo_spend:
    spend_mode = st.sidebar.radio("Weighting Mode", ["Population-based", "Equal", "Custom"], index=0, horizontal=True, key='spend_mode', help="How to distribute sample size across active regions.")

if calculate_geo_spend:
    with st.expander("Configure Active Regions and Custom Data", expanded=False):
        with st.form("region_selection_form"):
            temp_selections = []
            cols = st.columns(3)
            for i, region in enumerate(ALL_REGIONS):
                with cols[i % 3]:
                    if st.checkbox(region, value=(region in st.session_state.get('selected_regions', ALL_REGIONS)), key=f"check_{region}"):
                        temp_selections.append(region)
            submitted = st.form_submit_button("Confirm Region Selection")
            if submitted:
                st.session_state.selected_regions = temp_selections
                st.session_state["custom_geo_data_editor"] = GEO_DEFAULTS[GEO_DEFAULTS['Region'].isin(st.session_state.selected_regions)].copy()
                st.rerun()
        if spend_mode == 'Custom':
            st.markdown("---")
            st.write("Edit weights and CPMs below. Your edits will be saved automatically.")
            if "custom_geo_data_editor" not in st.session_state:
                st.session_state["custom_geo_data_editor"] = GEO_DEFAULTS[GEO_DEFAULTS['Region'].isin(st.session_state.get('selected_regions', ALL_REGIONS))].copy()
            edited_df = st.data_editor(st.session_state["custom_geo_data_editor"], num_rows="dynamic", use_container_width=True, key="custom_geo_data_editor")
            current_sum = edited_df['Weight'].sum()
            st.metric(label="Current Weight Sum", value=f"{current_sum:.2%}", delta=f"{(current_sum - 1.0):.2%} from target")
            if not np.isclose(current_sum, 1.0): st.warning("Sum of weights must be 100%.")

st.markdown("---")
submit = st.sidebar.button("Run Calculation", type="primary", use_container_width=True)

if submit:
    st.header("Results")
    req_n, total_spend, weeks = None, None, None
    num_groups = 1 + num_variants
    
    if mode == "Estimate Sample Size":
        if methodology == "Frequentist": req_n = calculate_sample_size_frequentist(p_A, uplift, desired_power, alpha, num_variants)
        else:
            b_results = simulate_power(p_A, uplift, desired_power, sims, samples, 1, 1, num_variants)
            if b_results: req_n = b_results[0][0]
    else: req_n = fixed_n
    
    total_users = req_n * num_groups if req_n else 0
    
    if calculate_geo_spend and req_n:
        if st.session_state.get('selected_regions', ALL_REGIONS):
            geo_df = pd.DataFrame()
            if spend_mode == "Custom":
                geo_df = pd.DataFrame(st.session_state.get("custom_geo_data_editor", []))
                if geo_df.empty: geo_df = GEO_DEFAULTS[GEO_DEFAULTS['Region'].isin(st.session_state.get('selected_regions', ALL_REGIONS))].copy()
                if not np.isclose(geo_df['Weight'].sum(), 1.0):
                    st.error("Final check failed: Custom weights must sum to 1.0."); geo_df = pd.DataFrame()
            else:
                base_df = GEO_DEFAULTS[GEO_DEFAULTS['Region'].isin(st.session_state.get('selected_regions', ALL_REGIONS))].copy()
                if not base_df.empty:
                    if spend_mode == "Population-based": base_df["Weight"] /= base_df["Weight"].sum()
                    else: base_df["Weight"] = 1 / len(base_df)
                geo_df = base_df
            
            if not geo_df.empty:
                geo_df["Users"] = (geo_df["Weight"] * total_users).astype(int)
                geo_df["Impressions (k)"] = geo_df["Users"] / 1000
                geo_df["Spend (£)"] = geo_df["Impressions (k)"] * geo_df["CPM (£)"]
                total_spend = geo_df['Spend (£)'].sum()

    if weekly_traffic > 0 and req_n:
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
        mde_results = None
        if methodology == "Frequentist":
            mde_results = calculate_mde_frequentist(p_A, fixed_n, desired_power, alpha, num_variants)
        else: # Bayesian
            mde_results = simulate_mde_bayesian(p_A, fixed_n, desired_power, sims, samples, 1, 1, num_variants)
        
        if mde_results and mde_results[-1][1] >= desired_power:
            mde, achieved_power = mde_results[-1]
            st.success(f"With **{fixed_n:,} users** per group, the smallest uplift you can reliably detect is **{mde:.2%}** (with {achieved_power:.1%} power).")
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
            sizes = [calculate_sample_size_frequentist(p_A, u, desired_power, alpha, num_variants) for u in uplifts_plot if u > 0 and p_A*(1+u) <= 1]
            valid_uplifts = [u for u in uplifts_plot if u > 0 and p_A*(1+u) <= 1 and calculate_sample_size_frequentist(p_A, u, desired_power, alpha, num_variants) is not None]
            sizes = [s for s in sizes if s is not None]
            if valid_uplifts:
                fig2, ax2 = plt.subplots()
                ax2.plot([u * 100 for u in valid_uplifts], sizes); ax2.axvline(uplift * 100, linestyle='--', color='red', label=f"Your Target ({uplift:.1%})")
                ax2.set_xlabel("Uplift (%)"); ax2.set_ylabel("Sample Size per Group")
                ax2.set_title("Sample Size vs Uplift"); ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6); st.pyplot(fig2)

else:
    st.info("Set your parameters in the sidebar and click 'Run Calculation'.")
