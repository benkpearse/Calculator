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

# --- Bayesian Functions ---
@st.cache_data
def run_simulation(n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh):
    """
    Runs a single set of simulations for a given sample size and conversion rates.
    Returns the calculated power.
    """
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
    """
    Simulates Bayesian power across a range of sample sizes.
    """
    p_B = p_A * (1 + uplift)
    if p_B > 1.0:
        st.error(f"Error: Uplift of {uplift:.2%} on baseline {p_A:.2%} results in a conversion rate > 100%.")
        return []

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
    """
    Simulates Bayesian MDE for a fixed sample size.
    """
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

# --- Frequentist Function ---
@st.cache_data
def calculate_sample_size_frequentist(p_A, uplift, power=0.8, alpha=0.05):
    """
    Calculates sample size using a two-sided Z-test formula.
    """
    p_B = p_A * (1 + uplift)
    if p_B > 1.0:
        # This error is better handled in the UI, but good to have a safeguard
        return None
        
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    # Pooled variance for the numerator
    p_pooled = (p_A + p_B) / 2
    numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) + z_beta * np.sqrt(p_A * (1 - p_A) + p_B * (1 - p_B)))**2
    denominator = (p_A - p_B) ** 2
    sample_size_per_group = numerator / denominator
    return int(np.ceil(sample_size_per_group))

# --- Geo Testing Data ---
UK_REGIONS = [
    "North East", "North West", "Yorkshire and the Humber", "East Midlands",
    "West Midlands", "East of England", "London", "South East",
    "South West", "Wales", "Scotland", "Northern Ireland"
]
DEFAULT_CPMS = [7.50, 8.00, 8.25, 7.00, 7.80, 8.10, 12.00, 10.00, 7.60, 6.90, 9.00, 8.50]
DEFAULT_WEIGHTS = [0.03, 0.09, 0.07, 0.07, 0.09, 0.10, 0.18, 0.16, 0.07, 0.04, 0.07, 0.03]


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
    
    p_A = st.number_input(
        "Baseline conversion rate (p_A)", min_value=0.0001, max_value=0.999, value=0.05, step=0.001,
        format="%.4f", help="Conversion rate for your control variant (A)."
    )
    uplift = st.number_input(
        "Expected uplift", min_value=0.0001, max_value=0.999, value=0.10, step=0.01,
        format="%.4f", help="Relative improvement expected in the variant (e.g., 0.10 for +10%)."
    )

    # --- Conditional Inputs based on Methodology ---
    if methodology == "Bayesian":
        st.subheader("Bayesian Settings")
        thresh = st.slider(
            "Posterior threshold", 0.80, 0.99, 0.95, step=0.01,
            help="The P(B > A) threshold required to declare a winner. 95% is common."
        )
        desired_power = st.slider(
            "Desired power", 0.5, 0.99, 0.8, step=0.01,
            help="The probability of detecting the uplift if it's real. 80% is common."
        )
        simulations = st.slider("Simulations", 100, 5000, 500, step=100)
        samples = st.slider("Posterior samples", 500, 5000, 1000, step=100)
        
        st.subheader("Optional Priors")
        use_auto_prior = st.checkbox("Calculate priors from historical data")
        if use_auto_prior:
            hist_conv = st.number_input("Historical Conversions", min_value=0, value=50, step=1)
            hist_n = st.number_input("Historical Users", min_value=1, value=1000, step=1)
            alpha_prior = hist_conv
            beta_prior = hist_n - hist_conv
        else:
            alpha_prior = st.number_input("Alpha (prior successes)", min_value=0.0, value=1.0, step=0.1)
            beta_prior = st.number_input("Beta (prior failures)", min_value=0.0, value=1.0, step=0.1)

    else: # Frequentist
        st.subheader("Frequentist Settings")
        alpha = st.slider(
            "Significance level (α)", 0.01, 0.10, 0.05, step=0.01,
            help="The probability of a false positive. 0.05 is standard for 95% confidence."
        )
        power = st.slider(
            "Desired power (1 - β)", 0.5, 0.99, 0.8, step=0.01,
            help="The probability of detecting the uplift if it's real. 80% is common."
        )

    st.header("3. Estimate Duration")
    weekly_traffic = st.number_input(
        "Estimated total weekly traffic",
        min_value=1, value=20000, step=100,
        help="Total users entering the experiment each week (before the 50/50 split)."
    )

    st.markdown("---")
    run_button = st.button("Run Calculation", type="primary", use_container_width=True)

# 4. Main Page for Displaying Outputs
st.markdown("---")

if run_button:
    if methodology == "Bayesian":
        # The Bayesian calculator still supports MDE mode, so we need the radio button.
        mode = st.radio("Planning Mode", ["Estimate Sample Size", "Estimate MDE"])
        if mode == "Estimate Sample Size":
            results = simulate_power(p_A, uplift, thresh, desired_power, simulations, samples, alpha_prior, beta_prior)
            if results:
                x_vals, y_vals = zip(*results)
                required_sample_size = x_vals[-1] if y_vals[-1] >= desired_power else None
                st.subheader("📈 Bayesian Sample Size Estimation")
                if required_sample_size:
                    st.success(f"✅ Estimated minimum sample size per group: **{required_sample_size:,}** (achieved {y_vals[-1]:.1%} power).")
                else:
                    st.warning("Could not reach desired power. The uplift may be too small or the power target too high for a practical test.")
                
                # Visualization
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(x_vals, y_vals, marker='o', label='Estimated Power')
                ax.axhline(desired_power, color='red', linestyle='--', label='Target Power')
                ax.set_xlabel("Sample Size per Group")
                ax.set_xscale('log')
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
                ax.set_ylabel("Estimated Power")
                ax.set_title("Power vs. Sample Size")
                ax.grid(True, which="both", ls="--")
                ax.legend()
                st.pyplot(fig)
        else: # Bayesian MDE Mode
            fixed_n = st.number_input("Fixed sample size per variant", 100)
            results = simulate_mde(p_A, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, fixed_n)
            # ... (MDE results display would go here)
    
    else: # Frequentist
        required_sample_size = calculate_sample_size_frequentist(p_A, uplift, power=power, alpha=alpha)
        st.subheader("📈 Frequentist Sample Size Estimation")
        if required_sample_size:
            st.success(f"✅ You need at least **{required_sample_size:,} users per variant** to detect a {uplift:.2%} uplift with {power:.0%} power and {1-alpha:.0%} confidence.")
            
            # --- GEO AD SPEND SECTION ---
            total_users_needed = required_sample_size * 2
            st.subheader("💰 Estimated Ad Spend for Geo Test")
            st.markdown("Use this section to estimate the ad spend required to reach the necessary sample size for a geo-based test.")

            selected_regions = st.multiselect("Select regions to include in spend calculation:", UK_REGIONS, default=UK_REGIONS)
            
            if selected_regions:
                filtered_indices = [UK_REGIONS.index(r) for r in selected_regions]
                filtered_cpms = [DEFAULT_CPMS[i] for i in filtered_indices]
                filtered_weights = [DEFAULT_WEIGHTS[i] for i in filtered_indices]

                weight_sum = sum(filtered_weights)
                normalized_weights = [w / weight_sum for w in filtered_weights]

                spend_df = pd.DataFrame({
                    "Region": selected_regions,
                    "Weight": normalized_weights,
                    "CPM (£)": filtered_cpms
                })

                edited_df = st.data_editor(
                    spend_df, use_container_width=True, num_rows="dynamic",
                    column_config={
                        "Weight": st.column_config.ProgressColumn("Weight", format="%.2f", min_value=0, max_value=1),
                        "CPM (£)": st.column_config.NumberColumn(format="£%.2f", min_value=0.0, step=0.1)
                    }
                )

                if not np.isclose(edited_df["Weight"].sum(), 1.0):
                    st.error("❌ Region weights must sum to 1. Please adjust the values.")
                else:
                    region_costs = []
                    for _, row in edited_df.iterrows():
                        region_users = total_users_needed * row["Weight"]
                        region_impressions_in_thousands = region_users / 1000
                        region_cost = region_impressions_in_thousands * row["CPM (£)"]
                        region_costs.append(region_cost)
                    
                    edited_df["Estimated Spend (£)"] = region_costs
                    total_spend = edited_df["Estimated Spend (£)"].sum()
                    st.metric("Total Estimated Ad Spend", f"£{total_spend:,.2f}")

                    st.subheader("📊 Regional Spend Breakdown")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.barh(edited_df["Region"], edited_df["Estimated Spend (£)"])
                    ax.set_xlabel("Spend (£)")
                    ax.set_title("Estimated Ad Spend by UK Region")
                    st.pyplot(fig)

            # --- Sample Size vs Uplift Visualization ---
            st.subheader("🔬 Sample Size vs. Uplift")
            uplifts_range = np.linspace(uplift * 0.2, uplift * 2, 50)
            sample_sizes_range = [calculate_sample_size_frequentist(p_A, u, power=power, alpha=alpha) for u in uplifts_range if u > 0]
            valid_uplifts = [u for u in uplifts_range if u > 0]

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot([u * 100 for u in valid_uplifts], sample_sizes_range, color='blue')
            ax2.axvline(x=uplift * 100, linestyle='--', color='red', label=f'Your Target ({uplift:.1%})')
            ax2.set_xlabel("Minimum Detectable Uplift (%)")
            ax2.set_ylabel("Required Sample Size per Group")
            ax2.set_title("Sample Size vs. Uplift")
            ax2.grid(True)
            ax2.legend()
            st.pyplot(fig2)

    # --- Time-Based Planning (works for both) ---
    if 'required_sample_size' in locals() and required_sample_size:
        st.subheader("🗓️ Estimated Test Duration")
        users_per_week_per_variant = weekly_traffic / 2
        if users_per_week_per_variant > 0:
            estimated_weeks = required_sample_size / users_per_week_per_variant
            st.info(f"With {weekly_traffic:,} total users per week, you'll need approximately **{estimated_weeks:.1f} weeks** to reach the required sample size.")
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Calculation' to see the results.")

# 5. Explanations Section
st.markdown("---")
with st.expander("ℹ️ About the Methodologies"):
    st.markdown("""
    #### Bayesian vs. Frequentist Approaches
    This tool offers two different statistical philosophies for power analysis.

    **1. Bayesian (Simulation-Based)**
    - **What it is:** A modern approach that uses simulation to answer the question: *"If the true uplift is X%, what is the probability that our test will conclude that the variant is better?"*
    - **Pros:** More intuitive, flexible, and allows for the incorporation of prior knowledge (`alpha` and `beta` priors) from past experiments to make the analysis more data-efficient.
    - **Use when:** You want a more nuanced view of risk and probability, or when you have historical data to inform your assumptions.

    **2. Frequentist (Formula-Based)**
    - **What it is:** The traditional method taught in most statistics courses. It uses a mathematical formula based on a two-sided Z-test to calculate the sample size needed to achieve a desired level of statistical significance (`alpha`) and power (`1-beta`).
    - **Pros:** Very fast, deterministic (always gives the same answer), and widely understood.
    - **Use when:** You need a quick, standard calculation or when your organization's standard is to use p-values and significance testing.
    
    ---
    #### About Geo Testing Ad Spend
    Geo testing is used for channels where you can't randomize individual users, such as TV, radio, or print advertising. Instead, you randomize by geographic region (e.g., showing an ad in London but not in Manchester).
    - **How it works:** This calculator takes the total sample size required for your test and distributes it across different UK regions based on their population weights.
    - **Editable CPMs:** It uses default Cost Per Mille (CPM, cost per 1000 impressions) values for each region, which you can edit in the table to match your media plan.
    - **The Output:** The final table and chart show the estimated ad spend required in each region to acquire the necessary number of users to run a properly powered test.
    """)
