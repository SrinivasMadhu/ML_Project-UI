import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="AutoValuate AI | Regression, Classification & Clustering Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Dark Theme Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0d1527 100%);
        color: #f3f4f6;
    }
    
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.4);
    }
    
    .prediction-price {
        font-size: 2.7rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 8px 0;
    }

    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Caching Data & Artifacts ---
@st.cache_resource
def load_all_artifacts():
    base_dir = Path(__file__).resolve().parent

    # 1. Regression Model
    reg_model = joblib.load(base_dir / "car_price_model.pkl")
    car_df = pd.read_csv(base_dir / "car data.csv")
    car_df['Car_Age'] = 2026 - car_df['Year']
    car_df['Brand'] = car_df['Car_Name'].astype(str).str.split().str[0]

    # 2. Classification Model & Meta
    clf_model = joblib.load(base_dir / "loan_default_model.pkl")
    loan_features = joblib.load(base_dir / "loan_feature_names.pkl")
    loan_df = pd.read_csv(base_dir / "train copy.csv")

    # 3. Model Comparisons data
    clf_comparison_path = base_dir / "classification_all_10_models.csv"
    clf_comparison_df = pd.read_csv(clf_comparison_path) if clf_comparison_path.exists() else None

    # 4. Customer Personality Clustering Models & Data
    cust_prep = joblib.load(base_dir / "customer_preprocessor.pkl")
    cust_kmeans = joblib.load(base_dir / "customer_kmeans.pkl")
    cust_pca = joblib.load(base_dir / "customer_pca.pkl")
    cust_features = joblib.load(base_dir / "customer_feature_names.pkl")
    cust_df = pd.read_csv(base_dir / "customer_personality_cluster_assignments.csv")
    clustering_metrics_df = pd.read_csv(base_dir / "clustering_metrics.csv")

    return (reg_model, car_df, clf_model, loan_features, loan_df, clf_comparison_df,
            cust_prep, cust_kmeans, cust_pca, cust_features, cust_df, clustering_metrics_df)

(reg_model, car_df, clf_model, loan_features, loan_df, clf_comparison_df,
 cust_prep, cust_kmeans, cust_pca, cust_features, cust_df, clustering_metrics_df) = load_all_artifacts()

# Customer Cluster Profiles dictionary
CUSTOMER_CLUSTERS = {
    0: {"name": "Luxury & Connoisseur", "color": "#8b5cf6", "desc": "High income bracket with top spend on premium wines, meats, and gourmet products."},
    1: {"name": "Digital Active Shoppers", "color": "#38bdf8", "desc": "Moderate-high income with high engagement in digital channels and web store discounts."},
    2: {"name": "High-Net-Worth In-Store Elites", "color": "#10b981", "desc": "Highest average income ($77.5k+) with high physical store purchases and catalogue sales."},
    3: {"name": "Budget & Economical Families", "color": "#f59e0b", "desc": "Price-sensitive segment with lowest discretionary spending and reliance on bargain deals."}
}

# --- Top Hero Banner ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧠 Applied ML Suite: Regression, Classification & Clustering</div>
    <p class="hero-subtitle">Interactive Review 2 Intelligence Dashboard: Car Price Valuation, Loan Default Risk & Customer Personality Segmentation</p>
</div>
""", unsafe_allow_html=True)

# --- Global Top KPI Metrics ---
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown('<div class="metric-card"><div class="metric-val">0.965 R²</div><div class="metric-lbl">Regression (Car Price RF)</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="metric-card"><div class="metric-val">78.8% Acc</div><div class="metric-lbl">Classification (Loan Default)</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="metric-card"><div class="metric-val">4 Clusters</div><div class="metric-lbl">Customer Segments (K-Means)</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{len(cust_df)}</div><div class="metric-lbl">Customer Records Clustered</div></div>', unsafe_allow_html=True)

st.write("")

# --- Main App Navigation ---
tab_reg, tab_clf, tab_cluster, tab_benchmarks = st.tabs([
    "📈 Car Price Prediction (Regression)",
    "🛡️ Vehicle Loan Default Risk (Classification)",
    "👥 Customer Personality Segmentation (Clustering)",
    "📊 Benchmarks & Model Evaluation"
])

# =====================================================================
# TAB 1: REGRESSION (CAR SELLING PRICE)
# =====================================================================
with tab_reg:
    st.markdown("### 🚘 Used Car Valuation Estimator (Regression)")
    st.write("Input vehicle parameters to estimate current fair market resale value using the trained **Random Forest Regressor**.")

    col1, col2 = st.columns([1, 1], gap="large")
    unique_brands = sorted(car_df['Brand'].unique().tolist())
    default_brand = 'city' if 'city' in unique_brands else unique_brands[0]

    with col1:
        st.subheader("Vehicle Identity")
        selected_brand = st.selectbox("Vehicle Brand / Series", unique_brands, index=unique_brands.index(default_brand))
        year = st.slider("Year of Manufacture", min_value=2003, max_value=2024, value=2018, step=1)
        car_age = 2026 - year
        st.caption(f"Estimated vehicle age: **{car_age} years** (computed relative to 2026)")

        present_price = st.number_input(
            "Original / Current Showroom Ex-Showroom Price (in ₹ Lakhs)",
            min_value=0.3,
            max_value=100.0,
            value=9.54,
            step=0.25,
            help="1 Lakh = 100,000 INR."
        )

    with col2:
        st.subheader("Usage & Condition")
        kms_driven = st.number_input(
            "Odometer Reading (Kilometers Driven)",
            min_value=500,
            max_value=600000,
            value=35000,
            step=2500
        )
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            fuel_type = st.selectbox("Fuel Type", sorted(car_df['Fuel_Type'].unique().tolist()), index=sorted(car_df['Fuel_Type'].unique().tolist()).index("Petrol") if "Petrol" in car_df['Fuel_Type'].unique() else 0)
            seller_type = st.selectbox("Seller Type", sorted(car_df['Seller_Type'].unique().tolist()), index=0)
        with c_sub2:
            transmission = st.selectbox("Transmission", sorted(car_df['Transmission'].unique().tolist()), index=0)
            owner = st.selectbox("Previous Owners", options=[0, 1, 3], format_func=lambda x: f"{x} Prior Owner(s)" if x > 0 else "First Owner (0 Prior)", index=0)

    st.write("")
    if st.button("🚀 Calculate Estimated Selling Price", use_container_width=True, type="primary"):
        input_car = pd.DataFrame([{
            'Present_Price': float(present_price),
            'Kms_Driven': int(kms_driven),
            'Fuel_Type': fuel_type,
            'Seller_Type': seller_type,
            'Transmission': transmission,
            'Owner': int(owner),
            'Car_Age': int(car_age),
            'Brand': selected_brand
        }])

        pred_price = reg_model.predict(input_car)[0]
        pred_price_bounded = max(0.1, pred_price)
        mae_est = 0.62
        lower_bound = max(0.05, pred_price_bounded - mae_est)
        upper_bound = pred_price_bounded + mae_est

        st.markdown("---")
        r_c1, r_c2 = st.columns([1, 1], gap="medium")

        with r_c1:
            st.markdown(f"""
            <div class="prediction-card">
                <div style="font-size: 0.95rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.08em;">Estimated Resale Price</div>
                <div class="prediction-price">₹ {pred_price_bounded:.2f} Lakhs</div>
                <div style="color: #94a3b8; font-size: 0.95rem;">
                    Confidence Interval (±MAE): <b>₹ {lower_bound:.2f} L</b> — <b>₹ {upper_bound:.2f} L</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r_c2:
            retention_rate = (pred_price_bounded / present_price) * 100 if present_price > 0 else 0
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=min(100.0, retention_rate),
                number={'suffix': "%", 'font': {'size': 24, 'color': '#38bdf8'}},
                title={'text': "Value Retention vs Showroom", 'font': {'size': 14, 'color': '#94a3b8'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#38bdf8"},
                    'bgcolor': "#1e293b",
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(239, 68, 68, 0.25)"},
                        {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.25)"},
                        {'range': [70, 100], 'color': "rgba(16, 185, 129, 0.25)"}
                    ],
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#f3f4f6'},
                height=180,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_gauge, width='stretch')

# =====================================================================
# TAB 2: CLASSIFICATION (LOAN DEFAULT RISK)
# =====================================================================
with tab_clf:
    st.markdown("### 🛡️ Vehicle Loan Default Risk Prediction (Classification)")
    st.write("Predict whether a loan applicant is likely to default on vehicle financing using the trained **Classifier Pipeline**.")

    col_l1, col_l2 = st.columns([1, 1], gap="large")

    with col_l1:
        st.subheader("Financial & Asset Profile")
        disbursed_amount = st.number_input("Disbursed Amount (₹)", min_value=10000, max_value=1000000, value=55000, step=2500)
        asset_cost = st.number_input("Asset Total Cost (₹)", min_value=10000, max_value=2000000, value=75000, step=2500)
        ltv = (disbursed_amount / asset_cost * 100) if asset_cost > 0 else 75.0
        st.caption(f"Calculated Loan-to-Value (LTV): **{ltv:.2f}%**")

        employment_type = st.selectbox("Employment Type", ["Salaried", "Self employed", "Missing"])
        cns_score = st.slider("Bureau Credit Score (PERFORM_CNS_SCORE)", min_value=0, max_value=900, value=650, step=10,
                              help="0 indicates No History / Inactive. 300-900 indicates active credit history.")

    with col_l2:
        st.subheader("Applicant History & Verification")
        cns_desc = st.selectbox(
            "Credit Bureau Score Description",
            sorted(loan_df['PERFORM_CNS_SCORE_DESCRIPTION'].dropna().unique().tolist())
        )
        
        pri_sanctioned = st.number_input("Primary Sanctioned Amount (₹)", min_value=0, max_value=5000000, value=50000, step=5000)
        pri_disbursed = st.number_input("Primary Disbursed Amount (₹)", min_value=0, max_value=5000000, value=50000, step=5000)

        c_doc1, c_doc2 = st.columns(2)
        with c_doc1:
            aadhar_flag = st.checkbox("Aadhar Card Verified", value=True)
            pan_flag = st.checkbox("PAN Card Verified", value=False)
        with c_doc2:
            voter_flag = st.checkbox("Voter ID Verified", value=False)
            driving_flag = st.checkbox("Driving License Verified", value=False)

    st.write("")
    if st.button("🔍 Assess Loan Default Risk", use_container_width=True, type="primary"):
        sample_row = {}
        for col in loan_features:
            if col in loan_df.columns:
                if pd.api.types.is_numeric_dtype(loan_df[col]):
                    sample_row[col] = float(loan_df[col].median())
                else:
                    sample_row[col] = str(loan_df[col].mode().iloc[0]) if not loan_df[col].mode().empty else "Unknown"

        sample_row['DISBURSED_AMOUNT'] = float(disbursed_amount)
        sample_row['ASSET_COST'] = float(asset_cost)
        sample_row['LTV'] = float(ltv)
        sample_row['EMPLOYMENT_TYPE'] = employment_type
        sample_row['PERFORM_CNS_SCORE'] = float(cns_score)
        sample_row['PERFORM_CNS_SCORE_DESCRIPTION'] = cns_desc
        sample_row['PRI_SANCTIONED_AMOUNT'] = float(pri_sanctioned)
        sample_row['PRI_DISBURSED_AMOUNT'] = float(pri_disbursed)
        sample_row['AADHAR_FLAG'] = int(aadhar_flag)
        sample_row['PAN_FLAG'] = int(pan_flag)
        sample_row['VOTERID_FLAG'] = int(voter_flag)
        sample_row['DRIVING_FLAG'] = int(driving_flag)

        input_loan_df = pd.DataFrame([sample_row])

        pred_class = int(clf_model.predict(input_loan_df)[0])
        pred_proba = clf_model.predict_proba(input_loan_df)[0]
        default_prob = pred_proba[1] * 100

        st.markdown("---")
        res_l1, res_l2 = st.columns([1, 1], gap="medium")

        with res_l1:
            if pred_class == 0:
                st.markdown(f"""
                <div class="prediction-card" style="border-color: #10b981; background: rgba(16, 185, 129, 0.1);">
                    <div style="font-size: 0.95rem; color: #10b981; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700;">
                        Risk Level: LOW RISK
                    </div>
                    <div class="prediction-price" style="color: #10b981;">NO DEFAULT</div>
                    <div style="color: #cbd5e1; font-size: 0.95rem;">
                        Default Probability: <b>{default_prob:.1f}%</b>
                    </div>
                    <span class="status-badge" style="background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981;">
                        ✅ High Likelihood of Timely Repayment
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card" style="border-color: #ef4444; background: rgba(239, 68, 68, 0.1);">
                    <div style="font-size: 0.95rem; color: #ef4444; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700;">
                        Risk Level: HIGH RISK
                    </div>
                    <div class="prediction-price" style="color: #ef4444;">DEFAULT ALERT</div>
                    <div style="color: #cbd5e1; font-size: 0.95rem;">
                        Default Probability: <b>{default_prob:.1f}%</b>
                    </div>
                    <span class="status-badge" style="background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444;">
                        ⚠️ Additional Collateral or Verification Advised
                    </span>
                </div>
                """, unsafe_allow_html=True)

        with res_l2:
            fig_prob = go.Figure(go.Indicator(
                mode="gauge+number",
                value=default_prob,
                number={'suffix': "%", 'font': {'size': 24, 'color': '#ef4444' if default_prob > 50 else '#10b981'}},
                title={'text': "Default Probability Gauge", 'font': {'size': 14, 'color': '#94a3b8'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#94a3b8"},
                    'bar': {'color': '#ef4444' if default_prob > 50 else '#10b981'},
                    'bgcolor': "#1e293b",
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                        {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                    ],
                }
            ))
            fig_prob.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#f3f4f6'},
                height=180,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_prob, width='stretch')

# =====================================================================
# TAB 3: CLUSTERING (CUSTOMER PERSONALITY ANALYSIS)
# =====================================================================
with tab_cluster:
    st.markdown("### 👥 Customer Personality Segmentation (Clustering)")
    st.write("Unsupervised grouping using **K-Means (k=4)** and **Agglomerative Hierarchical Clustering** on customer spending behaviors.")

    # Section 1: Customer Profile Classification
    st.subheader("1. Instant Customer Segment Assignment")
    c_c1, c_c2 = st.columns([1, 1], gap="large")

    with c_c1:
        cust_income = st.number_input("Annual Household Income ($)", min_value=1000, max_value=250000, value=65000, step=2500)
        cust_education = st.selectbox("Education Level", ["Graduation", "PhD", "Master", "2n Cycle", "Basic"])
        cust_marital = st.selectbox("Marital Status", ["Married", "Together", "Single", "Divorced", "Widow"])
        cust_kidhome = st.slider("Number of Small Children at Home", 0, 3, 0)
        cust_teenhome = st.slider("Number of Teenagers at Home", 0, 3, 1)

    with c_c2:
        cust_wine = st.number_input("Annual Spend on Wines ($)", 0, 2500, 450, step=50)
        cust_meat = st.number_input("Annual Spend on Meat ($)", 0, 2000, 250, step=50)
        cust_gold = st.number_input("Annual Spend on Gold / Luxury ($)", 0, 500, 60, step=10)
        cust_web_purchases = st.slider("Web Store Purchases per Month", 0, 30, 6)
        cust_store_purchases = st.slider("In-Store Purchases per Month", 0, 30, 8)

    if st.button("🎯 Assign Customer Segment", use_container_width=True, type="primary"):
        # Build customer input row
        cust_sample = {}
        for col in cust_features:
            if col in cust_df.columns:
                if pd.api.types.is_numeric_dtype(cust_df[col]):
                    cust_sample[col] = float(cust_df[col].median())
                else:
                    cust_sample[col] = str(cust_df[col].mode().iloc[0]) if not cust_df[col].mode().empty else "Unknown"

        cust_sample['Income'] = float(cust_income)
        cust_sample['Education'] = cust_education
        cust_sample['Marital_Status'] = cust_marital
        cust_sample['Kidhome'] = int(cust_kidhome)
        cust_sample['Teenhome'] = int(cust_teenhome)
        cust_sample['MntWines'] = float(cust_wine)
        cust_sample['MntMeatProducts'] = float(cust_meat)
        cust_sample['MntGoldProds'] = float(cust_gold)
        cust_sample['NumWebPurchases'] = int(cust_web_purchases)
        cust_sample['NumStorePurchases'] = int(cust_store_purchases)

        cust_input_df = pd.DataFrame([cust_sample])
        cust_input_proc = cust_prep.transform(cust_input_df)
        pred_cust_cluster = int(cust_kmeans.predict(cust_input_proc)[0])
        cluster_info = CUSTOMER_CLUSTERS.get(pred_cust_cluster, {"name": f"Cluster {pred_cust_cluster}", "color": "#6366f1", "desc": "Custom Segment"})

        st.markdown(f"""
        <div class="prediction-card" style="border-color: {cluster_info['color']}; margin-top: 15px;">
            <div style="font-size: 0.95rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.08em;">Assigned Customer Personality Segment</div>
            <div class="prediction-price" style="color: {cluster_info['color']};">{cluster_info['name']}</div>
            <div style="color: #cbd5e1; font-size: 1rem; max-width: 700px; margin: 0 auto;">
                {cluster_info['desc']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("2. 2D PCA Cluster Map (Visualizing All Customers)")
    cust_plot_df = cust_df.copy()
    cust_plot_df['Segment_Name'] = cust_plot_df['KMeans_Cluster'].map(lambda c: CUSTOMER_CLUSTERS.get(c, {}).get('name', f'Cluster {c}'))

    fig_pca_cust = px.scatter(
        cust_plot_df,
        x='PCA1',
        y='PCA2',
        color='Segment_Name',
        hover_data=['Income', 'MntWines', 'MntMeatProducts', 'NumStorePurchases'],
        color_discrete_map={v['name']: v['color'] for v in CUSTOMER_CLUSTERS.values()},
        title="Customer Personality Clusters Projected on 2D PCA Space",
        labels={'PCA1': 'Principal Component 1 (Spending & Wealth)', 'PCA2': 'Principal Component 2 (Channels & Demographics)'}
    )
    fig_pca_cust.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(15, 23, 42, 0.6)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        font=dict(family="Plus Jakarta Sans", color="#e2e8f0"),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_pca_cust, width='stretch')

    # Section 3: Metric Comparison (KMeans vs Agglomerative)
    st.markdown("#### 3. Clustering Quality Metrics (K-Means vs. Agglomerative Hierarchical)")
    st.dataframe(clustering_metrics_df, width='stretch')

# =====================================================================
# TAB 4: BENCHMARKS & MODEL COMPARISONS
# =====================================================================
with tab_benchmarks:
    st.subheader("Model Evaluation & Review 2 Benchmarking")

    st.markdown("#### 1. Classification: All 10 Models Comparison")
    if clf_comparison_df is not None:
        fig_bar = px.bar(
            clf_comparison_df.sort_values('Accuracy', ascending=True),
            x='Accuracy',
            y='Model',
            orientation='h',
            color='F1_weighted',
            title="10 Classification Models Comparison (Accuracy & F1-Score)",
            color_continuous_scale='plasma'
        )
        fig_bar.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(15, 23, 42, 0.6)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            font=dict(family="Plus Jakarta Sans", color="#e2e8f0"),
            height=360
        )
        st.plotly_chart(fig_bar, width='stretch')
        st.dataframe(clf_comparison_df, width='stretch')
    else:
        st.info("Classification model comparison results loaded from project notebook.")

    st.markdown("---")
    st.markdown("#### 2. Regression: Actual vs Predicted (Car Selling Price)")
    from sklearn.model_selection import train_test_split
    feature_cols = ['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Car_Age', 'Brand']
    X_reg = car_df[feature_cols]
    y_reg = car_df['Selling_Price']
    _, X_test_reg, _, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    y_test_pred = reg_model.predict(X_test_reg)

    scatter_df = pd.DataFrame({'Actual': y_test_reg, 'Predicted': y_test_pred})
    fig_scatter = px.scatter(
        scatter_df,
        x='Actual',
        y='Predicted',
        trendline="ols",
        title="Goodness-of-Fit on Test Set (R² = 0.965)",
        labels={'Actual': 'Actual Price (₹ Lakhs)', 'Predicted': 'Predicted Price (₹ Lakhs)'},
        color_discrete_sequence=['#38bdf8']
    )
    fig_scatter.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(15, 23, 42, 0.6)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        font=dict(family="Plus Jakarta Sans", color="#e2e8f0"),
        height=360
    )
    st.plotly_chart(fig_scatter, width='stretch')

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 12px 0;">
    Applied ML Suite | Regression, Classification & Clustering | Streamlit & Scikit-Learn | Review 2 Ready
</div>
""", unsafe_allow_html=True)
