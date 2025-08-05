import streamlit as st
import numpy as np
from scipy.stats import beta, norm
import matplotlib.pyplot as plt
import pandas as pd

# 1. Set Page Configuration
st.set_page_config(
    page_title="Power Calculator | Bayesian Toolkit",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Core Calculation Functions ---

@st.cache_data
def run_simulation(n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh):
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
def simulate_power(p_A, uplift, thresh, desired_power, simulations, samples, alpha_prior, beta_prior):
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
def simulate_mde(p_A, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, fixed_n):
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
def calculate_sample_size_frequentist(p_A, uplift, power=0.8, alpha=0.05):
    p_B = p_A * (1 + uplift)
    if p_B >= 1: return None
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    p_pool = (p_A + p_B) / 2
    num = (z_alpha * np.sqrt(2 * p_pool * (1 - p_pool)) +
           z_beta * np.sqrt(p_A * (1 - p_A) + p_B * (1 - p_B))) ** 2
    denom = (p_B - p_A) ** 2
    return int(np.ceil(num / denom))

@st.cache_data
def calculate_mde_frequentist(p_A, n, power_target=0.8, alpha=0.05):
    results = []
    for uplift in np.linspace(0.001, 0.50, 100):
        p_B = p_A * (1 + uplift)
        if p_B > 1.0: continue
        se = np.sqrt(p_A * (1 - p_A) / n + p_B * (1 - p_B) / n)
        z_alpha = norm.ppf(1 - alpha / 2)
        power = norm.cdf((abs(p_B - p_A) / se) - z_alpha)
        results.append((uplift, power))
        if power >= power_target:
            break
    return results

# --- Geo Testing Data ---
UK_REGIONS = ["North East", "North West", "Yorkshire and the Humber", "East Midlands", "West Midlands", "East of England", "London", "South East", "South West", "Wales", "Scotland", "Northern Ireland"]
POP_WEIGHTS = [0.03, 0.09, 0.07, 0.07, 0.09, 0.10, 0.18, 0.16, 0.07, 0.04, 0.07, 0.03]
DEFAULT_CPMS = [7.50, 8.00, 8.25, 7.00, 7.80, 8.10, 12.00, 10.00, 7.60, 6.90, 9.00, 8.50]

# Build geo DataFrame
def build_geo_df(regions, weights, cpms):
    df = pd.DataFrame({"Region": regions, "Weight": weights, "CPM (£)": cpms})
    if df["Weight"].sum() > 0:
        df["Weight"] /= df["Weight"].sum()
    return df

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
    weekly_traffic = st.number_input("Weekly traffic", 1, 100000, 20000, help="Total users entering the experiment each week (before the 50/50 split).")
    calculate_geo_spend = st.checkbox("Calculate Geo Spend", value=False, help="Enable to plan ad spend for a geo-based test.")
    
    if calculate_geo_spend and methodology == "Frequentist":
        with st.expander("Configure Geo Spend Data"):
            spend_mode = st.radio("Weighting Mode", ["Population-based", "Equal", "Custom"], index=0, horizontal=True, help="Choose how to distribute the sample size across regions.")
            
            if spend_mode == "Custom":
                st.caption("Edit regional weights and CPMs as needed.")
                if 'geo_df_custom' not in st.session_state:
                    st.session_state.geo_df_custom = build_geo_df(UK_REGIONS, POP_WEIGHTS, DEFAULT_CPMS)
                
                edited_df = st.data_editor(st.session_state.geo_df_custom, num_rows="dynamic")
                st.session_state.geo_df_custom = edited_df

    submit = st.form_submit_button("Run Calculation")

if submit:
    st.header("Results")
    req_n = None
    
    if methodology == "Frequentist":
        if mode == "Estimate Sample Size":
            req_n = calculate_sample_size_frequentist(p_A, uplift, desired_power, alpha)
            st.subheader("📈 Required Sample Size")
            if req_n: st.success(f"**{req_n:,} per variant**")
            else: st.error("Unable to compute sample size. Uplift may be too high.")
        else: # MDE
            results = calculate_mde_frequentist(p_A, fixed_n, desired_power, alpha)
            req_n = fixed_n
            st.subheader("📉 Minimum Detectable Effect (MDE)")
            if results and results[-1][1] >= desired_power:
                mde, achieved_power = results[-1]
                st.success(f"**{mde:.2%}** (achieved {achieved_power:.1%} power)")
            else:
                st.warning("Could not reach desired power.")
    else: # Bayesian
        if mode == "Estimate Sample Size":
            results = simulate_power(p_A, uplift, thresh, desired_power, sims, samples, a0, b0)
            st.subheader("📈 Required Sample Size")
            if results and results[-1][1] >= desired_power:
                req_n = results[-1][0]
                st.success(f"**{req_n:,} per variant** (achieved {results[-1][1]:.1%} power)")
            else:
                req_n = None
                st.error("Could not reach desired power.")
        else: # MDE
            results = simulate_mde(p_A, thresh, desired_power, sims, samples, a0, b0, fixed_n)
            req_n = fixed_n
            st.subheader("📉 Minimum Detectable Effect (MDE)")
            if results and results[-1][1] >= desired_power:
                mde, achieved_power = results[-1]
                st.success(f"**{mde:.2%}** (achieved {achieved_power:.1%} power)")
            else:
                st.warning("Could not reach desired power.")

    if calculate_geo_spend and methodology == "Frequentist" and req_n:
        st.subheader("💰 Geo Ad Spend")
        if spend_mode == "Equal":
            geo_df = build_geo_df(UK_REGIONS, [1]*len(UK_REGIONS), DEFAULT_CPMS)
        elif spend_mode == "Population-based":
            geo_df = build_geo_df(UK_REGIONS, POP_WEIGHTS, DEFAULT_CPMS)
        else: # Custom
            geo_df = st.session_state.geo_df_custom
            if not np.isclose(geo_df["Weight"].sum(), 1.0):
                st.error("Custom weights must sum to 1.0. Please adjust in the sidebar.")
                geo_df = pd.DataFrame()

        if not geo_df.empty:
            total_users = req_n * 2
            geo_df["Users"] = geo_df["Weight"] * total_users
            geo_df["Impressions (k)"] = geo_df["Users"] / 1000
            geo_df["Spend (£)"] = geo_df["Impressions (k)"] * geo_df["CPM (£)"]
            st.dataframe(geo_df.style.format({"Weight":"{:.1%}","CPM (£)":"£{:.2f}","Spend (£)":"£{:,.2f}"}))
            st.download_button("Download CSV", geo_df.to_csv(index=False), file_name="geo_spend.csv")
            
            fig, ax = plt.subplots(figsize=(8,4))
            ax.barh(geo_df["Region"], geo_df["Spend (£)"])
            ax.set_xlabel("Spend (£)")
            ax.set_title("Geo Spend Breakdown")
            st.pyplot(fig)

    if req_n:
        st.subheader("🗓️ Estimated Test Duration")
        weeks = req_n / (weekly_traffic/2)
        st.info(f"You will need approximately **{weeks:.1f} weeks** to reach the required sample size.")

    if methodology == "Frequentist" and mode == "Estimate Sample Size" and req_n:
        st.subheader("🔬 Sample Size vs. Uplift")
        uplifts = np.linspace(uplift*0.5, uplift*1.5, 50)
        sizes = [calculate_sample_size_frequentist(p_A, u, desired_power, alpha) for u in uplifts if u > 0 and p_A*(1+u)<=1]
        valid_uplifts = [u for u in uplifts if u > 0 and p_A*(1+u)<=1]
        fig2, ax2 = plt.subplots()
        ax2.plot([u*100 for u in valid_uplifts], sizes)
        ax2.axvline(uplift*100, linestyle='--', color='red', label=f"Your Target ({uplift:.1%})")
        ax2.set_xlabel("Uplift (%)")
        ax2.set_ylabel("Sample Size per Variant")
        ax2.set_title("Sample Size vs Uplift")
        ax2.legend()
        st.pyplot(fig2)

else:
    st.info("Adjust the parameters in the sidebar and click 'Run Calculation'.")

# 5. Explanations Section
st.markdown("---")
with st.expander("ℹ️ About the Methodologies & Geo Testing"):
    st.markdown("""
    #### Bayesian vs. Frequentist Approaches
    - **Bayesian (Simulation-Based):** A modern approach that answers: *"What is the probability that my variant is better?"* It's intuitive and allows for the use of prior knowledge from past tests.
    - **Frequentist (Formula-Based):** The traditional method that uses p-values and significance levels (`alpha`). It's fast, deterministic, and widely understood.
    
    ---
    #### About Geo Testing Ad Spend
    Geo testing is for channels where you can't randomize individual users (e.g., TV, radio). Instead, you randomize by geographic region.
    - **How it works:** This calculator takes the total required sample size and distributes it across regions. You can use an equal split, a split based on population, or a fully custom split.
    - **Editable CPMs:** You can edit the Cost Per Mille (CPM) for each region in "Custom" mode to match your media plan.
    - **The Output:** The final table and chart show the estimated ad spend required in each region to run a properly powered test.
    """)
