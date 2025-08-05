import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

# --- Constants & Helpers ---
UK_REGIONS = [
    "North East", "North West", "Yorkshire and the Humber", "East Midlands",
    "West Midlands", "East of England", "London", "South East",
    "South West", "Wales", "Scotland", "Northern Ireland"
]
DEFAULT_CPMS = [7.50, 8.00, 8.25, 7.00, 7.80, 8.10, 12.00, 10.00, 7.60, 6.90, 9.00, 8.50]
HISTORICAL_PROFILE_KEY = "last_geo_profile"

@st.cache_data
def fetch_ons_weights():
    try:
        url = "https://example.com/ons_uk_region_weights.csv"
        df = pd.read_csv(url)
        df = df.set_index("Region").loc[UK_REGIONS].reset_index()
        return df["Weight"].tolist()
    except Exception:
        return [1/len(UK_REGIONS)] * len(UK_REGIONS)

@st.cache_data
def build_geo_df(regions, weights, cpms):
    df = pd.DataFrame({"Region": regions, "Weight": weights, "CPM (£)": cpms})
    total_weight = df["Weight"].sum()
    if total_weight > 0:
        df["Weight"] = df["Weight"] / total_weight
    return df

@st.cache_data
def calculate_power(p_A, p_B, n, alpha=0.05):
    se = np.sqrt(p_A * (1 - p_A) / n + p_B * (1 - p_B) / n)
    if se == 0:
        return 1.0
    z = norm.ppf(1 - alpha/2)
    effect_size = abs(p_B - p_A) / se
    return norm.cdf(effect_size - z) + norm.cdf(-effect_size - z)

@st.cache_data
def calculate_required_sample(p_A, uplift, power_target=0.8, alpha=0.05):
    p_B = p_A * (1 + uplift)
    if p_B >= 1:
        return None
    n = 100
    while n < 500000:
        power = calculate_power(p_A, p_B, n, alpha)
        if power >= power_target:
            return n
        n += 100
    return None

@st.cache_data
def calculate_mde(p_A, n, power_target=0.8, alpha=0.05):
    for uplift in np.linspace(0.001, 0.5, 100):
        p_B = p_A * (1 + uplift)
        if p_B >= 1: continue
        power = calculate_power(p_A, p_B, n, alpha)
        if power >= power_target:
            return uplift, power
    return None, 0

# --- UI Setup ---
st.set_page_config(page_title="Power Calculator", layout="centered")
st.title("⚙️ Pre-Test Power Calculator")

with st.sidebar.form("params_form"):
    st.header("1. Inputs")
    mode = st.radio("Goal", ["Estimate Sample Size", "Estimate MDE"])
    p_A = st.number_input("Baseline rate (p_A)", 0.0001, 0.9999, 0.05, 0.001)
    if mode == "Estimate Sample Size":
        uplift = st.number_input("Expected uplift", 0.0001, 0.999, 0.1, 0.01)
    else:
        fixed_n = st.number_input("Sample size per group", 100, 100000, 10000, 100)
    alpha = st.slider("Significance level (α)", 0.01, 0.1, 0.05)
    power_target = st.slider("Target Power (1-β)", 0.5, 0.99, 0.8)
    weekly_traffic = st.number_input("Weekly test traffic", 1, 1000000, 20000)

    st.header("2. Geo Spend")
    include_geo = st.checkbox("Include Geo Spend", value=False)
    geo_df = None
    if include_geo:
        preset = st.selectbox("Weight Preset", ["Equal", "ONS Population"], index=1)
        weights = fetch_ons_weights() if preset == "ONS Population" else [1/len(UK_REGIONS)] * len(UK_REGIONS)
        geo_df = build_geo_df(UK_REGIONS, weights, DEFAULT_CPMS)
        st.session_state[HISTORICAL_PROFILE_KEY] = geo_df

    submitted = st.form_submit_button("Run Calculation")

if submitted:
    if mode == "Estimate Sample Size":
        required_n = calculate_required_sample(p_A, uplift, power_target, alpha)
        if required_n:
            st.success(f"Required sample size per group: {required_n:,}")
            total_users = required_n * 2
        else:
            st.error("Could not calculate sample size.")
            total_users = 0
    else:
        uplift, power = calculate_mde(p_A, fixed_n, power_target, alpha)
        if uplift:
            st.success(f"Minimum detectable uplift: {uplift:.2%} with {power:.1%} power")
            total_users = fixed_n * 2
        else:
            st.error("Could not determine MDE.")
            total_users = 0

    if total_users:
        weeks = total_users / weekly_traffic
        st.info(f"Estimated test duration: {weeks:.1f} weeks")

    if include_geo and geo_df is not None and total_users > 0:
        geo_df = geo_df.copy()
        geo_df["Users"] = geo_df["Weight"] * total_users
        geo_df["Impressions (k)"] = geo_df["Users"] / 1000
        geo_df["Spend (£)"] = geo_df["Impressions (k)"] * geo_df["CPM (£)"]
        st.subheader("💰 Geo Spend Breakdown")
        st.dataframe(geo_df.style.format({"Weight":"{:.1%}","Spend (£)":"£{:.2f}"}))
        st.download_button("Download CSV", geo_df.to_csv(index=False), "geo_spend.csv")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.barh(geo_df['Region'], geo_df['Spend (£)'])
        ax.set_xlabel("Spend (£)")
        ax.set_title("Geo Spend by Region")
        st.pyplot(fig)
else:
    st.info("Set parameters and click 'Run Calculation'")
