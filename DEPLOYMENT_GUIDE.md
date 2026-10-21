# Review 2 Bonus: Regression, Classification & Clustering Multi-Task AI Suite

This project includes **all 3 Machine Learning paradigms** in a single interactive dashboard to secure the **+2 Bonus Marks**:
- **+1 Interactive GUI**: Streamlit web app with Regression, Classification, and Customer Clustering.
- **+1 Public Deployment**: Ready for one-click deployment on Streamlit Community Cloud or Hugging Face Spaces.

---

## 🌟 The 3 Integrated Machine Learning Pillars in `app.py`

### 1. 📈 Regression: Car Price Valuation
- **Notebook**: `Regression.ipynb`
- **Dataset**: `car data.csv`
- **Model**: Tuned Random Forest Regressor ($R^2 = 0.965$, $\text{MAE} \approx 0.62\text{ Lakhs}$).
- **Interactive Features**: Real-time resale price prediction (in ₹ Lakhs), dynamic confidence intervals, and showroom value retention gauge.

### 2. 🛡️ Classification: Vehicle Loan Default Risk
- **Notebook**: `classification_review2_vehicle_loan_default_corrected (1).ipynb`
- **Dataset**: `train copy.csv`
- **Model**: Random Forest Classifier Pipeline ($78.8\%$ Test Accuracy).
- **Interactive Features**: Risk classification badge (LOW RISK vs. HIGH RISK), continuous default probability gauge ($0 - 100\%$), and bureau verification analysis.

### 3. 👥 Clustering: Customer Personality Analysis
- **Notebook**: `clustering_review2_customer_personality.ipynb`
- **Dataset**: `marketing_campaign.csv`
- **Algorithms**: K-Means ($k=4$) and Agglomerative Hierarchical Clustering.
- **Interactive Features**:
  - **Instant Customer Segment Assignment**: Predicts which personality group a customer belongs to based on income, spend on wines/meats, and buying channels:
    - *Luxury & Connoisseur*
    - *Digital Active Shoppers*
    - *High-Net-Worth In-Store Elites*
    - *Budget & Economical Families*
  - **Interactive 2D PCA Cluster Map**: Plotly scatter plot projecting all 2,240 customers onto the first 2 principal components.
  - **Clustering Quality Metrics**: Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index for both K-Means and Agglomerative Clustering.

### 4. 📊 Evaluation & Benchmarking
- Interactive comparison of all 10 classification models (Accuracy vs. F1-Score).
- Actual vs. Predicted OLS regression fit chart.

---

## 🚀 How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit application
streamlit run app.py
```
Open your browser at: **`http://localhost:8501`**

---

## 🌐 Public Deployment (For the +1 Deployment Mark)

### Option A: Streamlit Community Cloud (Recommended)
1. Initialize git and commit:
   ```bash
   git init
   git add app.py requirements.txt "car data.csv" "train copy.csv" marketing_campaign.csv *.pkl *.csv
   git commit -m "Complete ML Suite: Regression, Classification, and Customer Clustering"
   git branch -M main
   git remote add origin https://github.com/<YOUR-USERNAME>/<REPO-NAME>.git
   git push -u origin main
   ```
2. Log in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Click **New app**, select your repository, set main file to `app.py`, and click **Deploy**.
4. You will get a permanent public URL: `https://<your-app>.streamlit.app`.

### Option B: Hugging Face Spaces
1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) with SDK set to **Streamlit**.
2. Push your code and files. The space will automatically build and provide a public URL.
