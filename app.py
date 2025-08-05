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
    if p_B > 1.0: return None
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    p_pooled = (p_A + p_B) / 2
    numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) + z_beta * np.sqrt(p_A * (1 - p_A) + p_B * (1 - p_B)))**2
    denominator = (p_A - p_B) ** 2
    sample_size_per_group = numerator / denominator
    return int(np.ceil(sample_size_per_group))

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
DEFAULT_CPMS = [7.50, 8.00, 8.25, 7.00, 7.80, 8.10, 12.00, 10.00, 7.60, 6.90, 9.00, 8.50]
POP_WEIGHTS = [0.03, 0.09, 0.07, 0.07, 0.09, 0.10, 0.18, 0.16, 0.07, 0.04, 0.07, 0.03]

# --- Refactored Display Functions ---
def display_bayesian_results(p_A, uplift, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, mode, fixed_n, weekly_traffic):
    # ... (Bayesian display logic remains the same) ...

def display_frequentist_results(p_A, uplift, power, alpha, weekly_traffic, calculate_geo_spend, mode, fixed_n, geo_df):
    if mode == "Estimate Sample Size":
        required_sample_size = calculate_sample_size_frequentist(p_A, uplift, power=power, alpha=alpha)
        st.subheader("📈 Frequentist Sample Size Estimation")
        if required_sample_size:
            st.success(f"✅ You need at least **{required_sample_size:,} users per variant** to detect a {uplift:.2%} uplift with {power:.0%} power and {1-alpha:.0%} confidence.")
            if calculate_geo_spend:
                display_geo_spend(required_sample_size, geo_df)
            
            st.subheader("🔬 Sample Size vs. Uplift")
            uplifts_range = np.linspace(uplift * 0.2, uplift * 2, 50)
            sample_sizes_range = [calculate_sample_size_frequentist(p_A, u, power=power, alpha=alpha) for u in uplifts_range if u > 0 and p_A * (1+u) <= 1]
            valid_uplifts = [u for u in uplifts_range if u > 0 and p_A * (1+u) <= 1]

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot([u * 100 for u in valid_uplifts], sample_sizes_range, color='blue')
            ax2.axvline(x=uplift * 100, linestyle='--', color='red', label=f'Your Target ({uplift:.1%})')
            ax2.set_xlabel("Minimum Detectable Uplift (%)")
            ax2.set_ylabel("Required Sample Size per Group")
            ax2.set_title("Sample Size vs. Uplift")
            ax2.grid(True)
            ax2.legend()
            st.pyplot(fig2)
            display_duration_estimate(required_sample_size, weekly_traffic)

    else: # MDE Mode
        results = calculate_mde_frequentist(p_A, fixed_n, power_target=power, alpha=alpha)
        st.subheader("📉 Frequentist MDE Estimation")
        if results and results[-1][1] >= power:
            mde, achieved_power = results[-1]
            st.success(f"✅ With **{fixed_n:,} users per group**, the minimum detectable uplift is **{mde:.2%}** (achieving {achieved_power:.1%} power).")
            display_duration_estimate(fixed_n, weekly_traffic)
        else:
            st.warning("Could not reach desired power with the given sample size. Try increasing the sample size.")

def display_duration_estimate(required_sample_size, weekly_traffic):
    st.subheader("🗓️ Estimated Test Duration")
    users_per_week_per_variant = weekly_traffic / 2
    if users_per_week_per_variant > 0:
        estimated_weeks = required_sample_size / users_per_week_per_variant
        st.info(f"With {weekly_traffic:,} total users per week, you'll need approximately **{estimated_weeks:.1f} weeks** to reach the required sample size.")

def display_geo_spend(required_sample_size, geo_df):
    total_users_needed = required_sample_size * 2
    st.subheader("💰 Estimated Ad Spend for Geo Test")
    
    if not geo_df.empty and np.isclose(geo_df["Weight"].sum(), 1.0):
        region_costs = []
        for _, row in geo_df.iterrows():
            region_users = total_users_needed * row["Weight"]
            region_impressions_in_thousands = region_users / 1000
            region_cost = region_impressions_in_thousands * row["CPM (£)"]
            region_costs.append(region_cost)
        
        geo_df["Estimated Spend (£)"] = region_costs
        total_spend = geo_df["Estimated Spend (£)"].sum()
        st.metric("Total Estimated Ad Spend", f"£{total_spend:,.2f}")

        st.subheader("📊 Regional Spend Breakdown")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(geo_df["Region"], geo_df["Estimated Spend (£)"])
        ax.set_xlabel("Spend (£)")
        ax.set_title("Estimated Ad Spend by UK Region")
        st.pyplot(fig)
    elif not geo_df.empty:
        st.error("❌ Region weights in the sidebar must sum to 1 to calculate spend.")


# 2. Page Title and Introduction
st.title("⚙️ Pre-Test Power Calculator")
st.markdown(
    "This tool helps you plan an A/B test by estimating the required sample size using either Bayesian or Frequentist methods."
)

# 3. Sidebar for All User Inputs
with st.sidebar:
    st.header("1. Choose Methodology")
    methodology = st.radio(
        "Calculation Method",
        ["Bayesian", "Frequentist"],
        horizontal=True,
        help="Choose 'Bayesian' for a simulation-based approach or 'Frequentist' for a traditional formula-based calculation."
    )
    
    st.header("2. Set Parameters")
    
    mode = st.radio("Planning Mode", ["Estimate Sample Size", "Estimate MDE"], horizontal=True)
    
    p_A = st.number_input("Baseline conversion rate (p_A)", 0.0001, 0.999, 0.05, 0.001, format="%.4f", help="Conversion rate for your control variant (A).")
    
    if mode == "Estimate Sample Size":
        uplift = st.number_input("Expected uplift", 0.0001, 0.999, 0.10, 0.01, format="%.4f", help="Relative improvement expected in the variant (e.g., 0.10 for +10%).")
    else: # MDE
        fixed_n = st.number_input("Fixed sample size per variant", 100, value=10000, step=100)

    if methodology == "Bayesian":
        st.subheader("Bayesian Settings")
        thresh = st.slider("Posterior threshold", 0.80, 0.99, 0.95, step=0.01, help="The P(B > A) threshold required to declare a winner. 95% is common.")
        desired_power = st.slider("Desired power", 0.5, 0.99, 0.8, step=0.01, help="The probability of detecting the uplift if it's real. 80% is common.")
        simulations = st.slider("Simulations", 100, 5000, 500, step=100)
        samples = st.slider("Posterior samples", 500, 5000, 1000, step=100)
        
        st.subheader("Optional Priors")
        use_auto_prior = st.checkbox("Calculate priors from historical data")
        if use_auto_prior:
            hist_conv = st.number_input("Historical Conversions", 0, value=50, step=1)
            hist_n = st.number_input("Historical Users", 1, value=1000, step=1)
            alpha_prior = hist_conv
            beta_prior = hist_n - hist_conv
        else:
            alpha_prior = st.number_input("Alpha (prior successes)", 0.0, value=1.0, step=0.1)
            beta_prior = st.number_input("Beta (prior failures)", 0.0, value=1.0, step=0.1)

    else: # Frequentist
        st.subheader("Frequentist Settings")
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, step=0.01, help="The probability of a false positive. 0.05 is standard for 95% confidence.")
        power = st.slider("Desired power (1 - β)", 0.5, 0.99, 0.8, step=0.01, help="The probability of detecting the uplift if it's real. 80% is common.")

    st.header("3. Estimate Duration")
    weekly_traffic = st.number_input("Estimated total weekly traffic", 1, value=20000, step=100, help="Total users entering the experiment each week (before the 50/50 split).")

    st.header("4. Geo Ad Spend (Optional)")
    calculate_geo_spend = st.checkbox("Calculate estimated ad spend for a Geo Test")
    
    geo_df = None
    if calculate_geo_spend and methodology == 'Frequentist':
        with st.expander("Configure Geo Spend Data"):
            spend_mode = st.radio("Weighting Mode", ["Population-based", "Equal", "Custom"], index=0, horizontal=True)
            
            if spend_mode == "Population-based":
                weights = POP_WEIGHTS
                geo_df = pd.DataFrame({"Region": UK_REGIONS, "Weight": weights, "CPM (£)": DEFAULT_CPMS})
                st.dataframe(geo_df, use_container_width=True)
            elif spend_mode == "Equal":
                weights = [1/len(UK_REGIONS)] * len(UK_REGIONS)
                geo_df = pd.DataFrame({"Region": UK_REGIONS, "Weight": weights, "CPM (£)": DEFAULT_CPMS})
                st.dataframe(geo_df, use_container_width=True)
            else: # Custom
                st.caption("Edit regional weights and CPMs as needed:")
                if 'geo_df_custom' not in st.session_state:
                    st.session_state.geo_df_custom = pd.DataFrame({
                        "Region": UK_REGIONS, "Weight": POP_WEIGHTS, "CPM (£)": DEFAULT_CPMS
                    })
                
                edited_df = st.data_editor(st.session_state.geo_df_custom, key='geo_df_editor',
                    column_config={"Weight": st.column_config.ProgressColumn("Weight", format="%.3f", min_value=0, max_value=1), "CPM (£)": st.column_config.NumberColumn(format="£%.2f", min_value=0.0, step=0.1)})
                st.session_state.geo_df_custom = edited_df
                geo_df = edited_df


    st.markdown("---")
    run_button = st.button("Run Calculation", type="primary", use_container_width=True)

# 4. Main Page for Displaying Outputs
st.markdown("---")

if run_button:
    if methodology == "Bayesian":
        mde_fixed_n = fixed_n if mode == 'Estimate MDE' else None
        bayesian_uplift = uplift if mode == 'Estimate Sample Size' else None
        display_bayesian_results(p_A, bayesian_uplift, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, mode, mde_fixed_n, weekly_traffic)
    else: # Frequentist
        freq_uplift = uplift if mode == 'Estimate Sample Size' else None
        freq_fixed_n = fixed_n if mode == 'Estimate MDE' else None
        display_frequentist_results(p_A, freq_uplift, power, alpha, weekly_traffic, calculate_geo_spend, mode, freq_fixed_n, geo_df)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Calculation' to see the results.")

# 5. Explanations Section
st.markdown("---")
with st.expander("ℹ️ About the Methodologies & Geo Testing"):
    st.markdown("""
    #### Bayesian vs. Frequentist Approaches
    This tool offers two different statistical philosophies for power analysis.

    **1. Bayesian (Simulation-Based)**
    - **What it is:** A modern approach that uses simulation to answer the question: *"If the true uplift is X%, what is the probability that our test will conclude that the variant is better?"*
    - **Pros:** More intuitive, flexible, and allows for the incorporation of prior knowledge.
    - **Use when:** You want a more nuanced view of risk and probability.

    **2. Frequentist (Formula-Based)**
    - **What it is:** The traditional method using a Z-test to calculate the sample size needed for a desired level of statistical significance (`alpha`) and power (`1-beta`).
    - **Pros:** Very fast, deterministic, and widely understood.
    - **Use when:** You need a quick, standard calculation.
    
    ---
    #### About Geo Testing Ad Spend
    Geo testing is used for channels where you can't randomize individual users (e.g., TV, radio). Instead, you randomize by geographic region.
    - **How it works:** This calculator takes the total required sample size and distributes it across regions based on their population weights.
    - **Editable CPMs:** You can edit the default Cost Per Mille (CPM) values in the sidebar to match your media plan.
    - **The Output:** The final table and chart show the estimated ad spend required in each region to acquire the necessary users for a properly powered test.
    """)
