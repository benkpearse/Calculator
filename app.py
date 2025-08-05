import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple

# 1. Set Page Configuration
st.set_page_config(
    page_title="A/B/n Test Power Calculator",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Core Calculation Functions (Frequentist Only) ---
@st.cache_data
def calculate_power_frequentist(p_A: float, p_B: float, n: int, alpha: float = 0.05, num_comparisons: int = 1) -> float:
    """Calculates power with Bonferroni correction for multiple comparisons."""
    if p_B < 0 or p_B > 1.0: return 0.0
    
    adjusted_alpha = alpha / num_comparisons
    
    se = np.sqrt(p_A * (1 - p_A) / n + p_B * (1 - p_B) / n)
    if se == 0: return 1.0
    
    effect_size_norm = abs(p_B - p_A) / se
    z_alpha = norm.ppf(1 - adjusted_alpha / 2)
    
    return norm.cdf(effect_size_norm - z_alpha) + norm.cdf(-effect_size_norm - z_alpha)

@st.cache_data
def calculate_sample_size_frequentist(p_A: float, uplift: float, power_target: float = 0.8, alpha: float = 0.05, num_variants: int = 1) -> int | None:
    """Calculates required sample size by iteratively searching for 'n'."""
    p_B = p_A * (1 + uplift)
    if p_B >= 1: return None
    n, power, MAX_SAMPLE_SIZE = 100, 0, 5_000_000
    with st.spinner("Calculating sample size..."):
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
    """Calculates MDE by iteratively searching for the uplift that meets the target power."""
    results = []
    with st.spinner("Calculating Minimum Detectable Effect..."):
        for uplift in np.linspace(0.001, 0.50, 100):
            p_B = p_A * (1 + uplift)
            if p_B > 1.0: continue
            power = calculate_power_frequentist(p_A, p_B, n, alpha, num_comparisons=num_variants)
            results.append((uplift, power))
            if power >= power_target:
                return results
    return []

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
    st.rerun()

# --- UI ---
st.title("⚙️ A/B/n Pre-Test Power Calculator")

st.markdown("""
<style>
    .bordered-container {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

with st.expander("What is Power Analysis? Click here to learn more.", expanded=False):
    st.markdown("""...""") # Content unchanged

st.sidebar.button("Reset All Settings", on_click=reset_app_state, use_container_width=True)
st.sidebar.markdown("---")

st.sidebar.header("1. Main Parameters")
num_variants = st.sidebar.number_input("Number of Variants (excluding control)", min_value=1, max_value=10, value=1, key='num_variants', help="An A/B test has 1 variant. An A/B/C test has 2 variants.")
mode = st.sidebar.radio("Planning Mode", ["Estimate Sample Size", "Estimate MDE"], horizontal=True, key='mode', help="Solve for sample size or minimum detectable effect.")
p_A = st.sidebar.number_input("Baseline rate (p_A)", 0.0001, 0.999, 0.05, 0.001, format="%.4f", key='p_A', help="Conversion rate of the control group.")
if mode == "Estimate Sample Size":
    uplift = st.sidebar.number_input("Expected uplift", 0.0001, 0.999, 0.10, 0.01, format="%.4f", key='uplift', help="Relative improvement you want to detect in the winning variant.")
else:
    fixed_n = st.sidebar.number_input("Fixed sample size per group", 100, value=10000, step=100, key='fixed_n', help="Users available for the control and EACH variant.")

st.sidebar.subheader("Test Settings")
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

# FIX: Combined the geo-spend logic into a single block to prevent NameError
if calculate_geo_spend:
    spend_mode = st.sidebar.radio("Weighting Mode", ["Population-based", "Equal", "Custom"], index=0, horizontal=True, key='spend_mode', help="How to distribute sample size across active regions.")
    
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
                st.session_state.custom_geo_df = GEO_DEFAULTS[GEO_DEFAULTS['Region'].isin(st.session_state.selected_regions)].copy()
                st.rerun()
        
        if spend_mode == 'Custom':
            st.markdown("---")
            st.write("Edit weights and CPMs below. Your edits will be saved automatically.")
            
            if 'custom_geo_df' not in st.session_state:
                st.session_state.custom_geo_df = GEO_DEFAULTS[GEO_DEFAULTS['Region'].isin(st.session_state.get('selected_regions', ALL_REGIONS))].copy()
            
            edited_df = st.data_editor(
                st.session_state.custom_geo_df, 
                num_rows="dynamic", 
                use_container_width=True,
                key="custom_geo_df_editor"
            )
            st.session_state.custom_geo_df = edited_df
            
            current_sum = edited_df['Weight'].sum()
            st.metric(label="Current Weight Sum", value=f"{current_sum:.2%}", delta=f"{(current_sum - 1.0):.2%} from target")
            if not np.isclose(current_sum, 1.0): st.warning("Sum of weights must be 100%.")

st.markdown("---")
submit = st.sidebar.button("Run Calculation", type="primary", use_container_width=True)

if 'submit' not in st.session_state:
    st.session_state.submit = False
if submit:
    st.session_state.submit = True

if st.session_state.submit:
    st.header("Results")
    req_n, total_spend, weeks = None, None, None
    num_groups = 1 + num_variants
    
    if mode == "Estimate Sample Size":
        req_n = calculate_sample_size_frequentist(p_A, uplift, desired_power, alpha, num_variants)
    else: 
        req_n = fixed_n
    
    total_users = req_n * num_groups if req_n else 0
    
    if calculate_geo_spend and req_n:
        selected_regions = st.session_state.get('selected_regions', ALL_REGIONS)
        if selected_regions:
            geo_df = pd.DataFrame()
            if spend_mode == "Custom":
                geo_df = st.session_state.get("custom_geo_df", pd.DataFrame()).copy()
                if not np.isclose(geo_df['Weight'].sum(), 1.0):
                    st.error("Final check failed: Custom weights must sum to 1.0."); geo_df = pd.DataFrame()
            else:
                base_df = GEO_DEFAULTS[GEO_DEFAULTS['Region'].isin(selected_regions)].copy()
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
        st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Could not determine the required sample size with the provided inputs.")

    if mode == "Estimate MDE":
        st.subheader("📉 Minimum Detectable Effect")
        mde_results = calculate_mde_frequentist(p_A, fixed_n, desired_power, alpha, num_variants)
        if mde_results:
            mde, achieved_power = mde_results[-1]
            st.success(f"With **{fixed_n:,} users** per group, the smallest uplift you can reliably detect is **{mde:.2%}** (with {achieved_power:.1%} power).")
        else:
            st.warning("Could not reach desired power within the tested uplift range (up to 50%). Please increase the sample size.")
    
    if calculate_geo_spend and total_spend is not None:
        with st.expander("View Geo Spend Breakdown"):
            st.write("**Spend Breakdown by Region**")
            style = {"Weight": "{:.1%}", "Users": "{:,.0f}", "CPM (£)": "£{:.2f}", "Impressions (k)": "{:,.1f}", "Spend (£)": "£{:,.2f}"}
            st.dataframe(geo_df.style.format(style), use_container_width=True)
            st.download_button("Download CSV", geo_df.to_csv(index=False), file_name="geo_spend_plan.csv")
            fig, ax = plt.subplots(); ax.barh(geo_df["Region"], geo_df["Spend (£)"])
            ax.set_xlabel("Spend (£)"); ax.set_title("Geo Spend Breakdown"); plt.tight_layout(); st.pyplot(fig)

    if mode == "Estimate Sample Size" and req_n:
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
