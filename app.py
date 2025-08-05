import streamlit as st
import numpy as np
from scipy.stats import beta, norm
import matplotlib.pyplot as plt
import pandas as pd

# --- Constants ---
UK_REGIONS = [
    "North East", "North West", "Yorkshire and the Humber", "East Midlands",
    "West Midlands", "East of England", "London", "South East",
    "South West", "Wales", "Scotland", "Northern Ireland"
]
# Population-based weights (ONS)
POP_WEIGHTS = [0.03, 0.09, 0.07, 0.07, 0.09, 0.10, 0.18, 0.16, 0.07, 0.04, 0.07, 0.03]
DEFAULT_CPMS = [7.50, 8.00, 8.25, 7.00, 7.80, 8.10, 12.00, 10.00, 7.60, 6.90, 9.00, 8.50]

# --- Helper Functions ---
@st.cache_data
def build_geo_df(regions, weights, cpms):
    df = pd.DataFrame({"Region": regions, "Weight": weights, "CPM (£)": cpms})
    df["Weight"] = df["Weight"]/df["Weight"].sum()
    return df

@st.cache_data
def calculate_sample_size_frequentist(p_A, uplift, power=0.8, alpha=0.05):
    p_B = p_A*(1+uplift)
    if p_B>=1: return None
    z_a = norm.ppf(1-alpha/2)
    z_b = norm.ppf(power)
    p_pool = (p_A+p_B)/2
    num = (z_a*np.sqrt(2*p_pool*(1-p_pool)) +
           z_b*np.sqrt(p_A*(1-p_A)+p_B*(1-p_B)))**2
    den = (p_B-p_A)**2
    return int(np.ceil(num/den))

@st.cache_data
def run_simulation(n, p_A, p_B, sims, samples, a0, b0, thresh):
    rng = np.random.default_rng(42)
    conv_A = rng.binomial(n, p_A, sims)
    conv_B = rng.binomial(n, p_B, sims)
    aA, bA = a0+conv_A, b0+n-conv_A
    aB, bB = a0+conv_B, b0+n-conv_B
    post_A = beta.rvs(aA, bA, size=(samples,sims), random_state=rng)
    post_B = beta.rvs(aB, bB, size=(samples,sims), random_state=rng)
    prob = np.mean(post_B>post_A, axis=0)
    return np.mean(prob>thresh)

@st.cache_data
def simulate_power(p_A, uplift, thresh, power, sims, samples, a0, b0):
    p_B = p_A*(1+uplift)
    if p_B>=1: return []
    res, n = [], 100
    while n<5_000_000:
        est = run_simulation(n,p_A,p_B,sims,samples,a0,b0,thresh)
        res.append((n,est))
        if est>=power: break
        n = n+100 if n<1000 else int(n*1.5) if n<20000 else int(n*1.25)
    return res

@st.cache_data
def simulate_mde(p_A, thresh, power, sims, samples, a0, b0, fixed_n):
    res=[]
    for uplift in np.linspace(0.01,0.5,50):
        p_B = p_A*(1+uplift)
        if p_B>=1: continue
        est=run_simulation(fixed_n,p_A,p_B,sims,samples,a0,b0,thresh)
        res.append((uplift,est))
        if est>=power: break
    return res

@st.cache_data
def calculate_mde_frequentist(p_A, n, power, alpha):
    for uplift in np.linspace(0.001,0.5,100):
        p_B=p_A*(1+uplift)
        if p_B>=1:continue
        se=np.sqrt(p_A*(1-p_A)/n+p_B*(1-p_B)/n)
        z_a=norm.ppf(1-alpha/2)
        pow=norm.cdf((abs(p_B-p_A)/se)-z_a)
        if pow>=power:
            return uplift,pow
    return None,None

# --- App UI ---
st.set_page_config(page_title="Power Calculator", layout="centered")
st.title("⚙️ Pre-Test Power Calculator")
st.markdown("Use Bayesian (simulation) or Frequentist (formula) then optional Geo Spend.")

# Sidebar form
def sidebar_form():
    with st.sidebar.form("params_form"):
        st.header("1. Method & Goal")
        methodology=st.radio("Methodology",["Bayesian","Frequentist"],horizontal=True)
        mode=st.radio("Goal",["Estimate Sample Size","Estimate MDE"],horizontal=True)
        st.header("2. Core Inputs")
        p_A=st.number_input("Baseline rate (p_A)",0.0001,0.999,0.05,0.001)
        if mode=="Estimate Sample Size":
            uplift=st.number_input("Expected uplift",0.0001,0.999,0.10,0.01)
        else:
            fixed_n=st.number_input("Fixed sample size per variant",100,100000,10000,100)
        if methodology=="Bayesian":
            thresh=st.slider("Posterior threshold",0.80,0.99,0.95)
            desired_power=st.slider("Desired power",0.5,0.99,0.8)
            sims=st.slider("Simulations",100,2000,500)
            samples=st.slider("Posterior samples",500,3000,1000)
            st.subheader("Optional Priors")
            if st.checkbox("Auto priors from history"):
                hist_c=st.number_input("Hist. successes",0,100,50)
                hist_n=st.number_input("Hist. users",1,10000,1000)
                a0,b0=hist_c,hist_n-hist_c
            else:
                a0=st.number_input("Alpha prior",1.0)
                b0=st.number_input("Beta prior",1.0)
        else:
            alpha=st.slider("Significance α",0.01,0.10,0.05)
            desired_power=st.slider("Desired power",0.5,0.99,0.8)
        st.header("3. Traffic & Duration")
        weekly=st.number_input("Weekly traffic",1,100000,20000,100)
        st.header("4. Geo Ad Spend")
        geo=st.checkbox("Include Geo Spend (Frequentist only)")
        if geo and methodology=="Frequentist":
            mode_geo=st.selectbox("Preset",["Population","Equal","Custom"])
            if mode_geo!="Custom":
                w=[1/len(UK_REGIONS)]*len(UK_REGIONS) if mode_geo=="Equal" else POP_WEIGHTS
                geo_df=build_geo_df(UK_REGIONS,w,DEFAULT_CPMS)
            else:
                init_df=build_geo_df(UK_REGIONS,POP_WEIGHTS,DEFAULT_CPMS)
                geo_df=st.data_editor(init_df,num_rows="fixed")
            st.session_state.geo_df=geo_df
        submit=st.form_submit_button("Run")
    return locals()

params=sidebar_form()
if params['submit']:
    # Unpack
    methodology=params['methodology']; mode=params['mode']; p_A=params['p_A']
    if mode=="Estimate Sample Size": uplift=params['uplift']
    else: fixed_n=params['fixed_n']
    if methodology=="Bayesian": thresh,desired_power,sims,samples,a0,b0=params['thresh'],params['desired_power'],params['sims'],params['samples'],params['a0'],params['b0']
    else: alpha,desired_power=params['alpha'],params['desired_power']
    weekly=params['weekly']; geo=params['geo']

    # Compute
    if methodology=="Frequentist":
        if mode=="Estimate Sample Size": req_n=calculate_sample_size_frequentist(p_A,uplift,desired_power,alpha)
        else:
            mde,ach=calculate_mde_frequentist(p_A,fixed_n,desired_power,alpha)
            req_n=fixed_n
    else:
        data=simulate_power(p_A,uplift,thresh,desired_power,sims,samples,a0,b0)
        req_n=data[-1][0] if data else None

    # Display
    if mode=="Estimate Sample Size":
        st.subheader("📈 Required Sample Size")
        if req_n: st.success(f"{req_n:,} per variant")
        else: st.error("Unable to compute.")
    else:
        st.subheader("📉 MDE")
        if methodology=="Frequentist": st.success(f"{mde:.2%} uplift detectable")
        else: st.write(data)

    # Geo Spend
    if geo and methodology=="Frequentist" and req_n and 'geo_df' in st.session_state:
        df=st.session_state.geo_df.copy()
        df['Users']=df['Weight']*(req_n*2)
        df['Impr(k)']=df['Users']/1000
        df['Spend(£)']=df['Impr(k)']*df['CPM (£)']
        st.subheader("💰 Geo Spend Breakdown")
        st.dataframe(df.style.format({"Weight":"{:.1%}","CPM (£)":"£{:.2f}","Spend(£)":"£{:,.2f}"}))
        st.download_button("Download CSV",df.to_csv(index=False),"geo_spend.csv")
        fig,ax=plt.subplots(figsize=(8,4));ax.barh(df['Region'],df['Spend(£)']);ax.set_xlabel('£');ax.set_title('Geo Spend');st.pyplot(fig)

    # Duration
    if req_n:
        weeks=req_n/(weekly/2)
        st.info(f"Estimated duration: {weeks:.1f} weeks")

    # Power curve
    if methodology=="Frequentist" and mode=="Estimate Sample Size" and req_n:
        uvals=np.linspace(uplift*0.5,uplift*1.5,50)
        svals=[calculate_sample_size_frequentist(p_A,u,desired_power,alpha) for u in uvals]
        fig2,ax2=plt.subplots();ax2.plot(uvals*100,svals);ax2.axvline(uplift*100,ls='--');ax2.set_xlabel('% Uplift');ax2.set_ylabel('Sample Size');st.pyplot(fig2)
else:
    st.info("Fill in parameters and click Run.")

# Explanations
st.markdown("---")
with st.expander("ℹ️ About methodologies & geo testing"):
    st.markdown(
        """
        **Bayesian (Simulation-Based)**: intuitive probability of detecting uplift, can use priors.
        
        **Frequentist (Formula-Based)**: fast, deterministic sample size & MDE via Z-test.
        
        **Geo Testing**: distributes required users across regions by selected weights & CPMs.
        """
    )
