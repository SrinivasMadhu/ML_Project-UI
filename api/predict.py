import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# Lazy load models for faster cold boot
_MODELS = {}

def get_models():
    if not _MODELS:
        _MODELS['reg'] = joblib.load(BASE_DIR / 'car_price_model.pkl')
        _MODELS['clf'] = joblib.load(BASE_DIR / 'loan_default_model.pkl')
        _MODELS['loan_features'] = joblib.load(BASE_DIR / 'loan_feature_names.pkl')
        _MODELS['cust_prep'] = joblib.load(BASE_DIR / 'customer_preprocessor.pkl')
        _MODELS['cust_kmeans'] = joblib.load(BASE_DIR / 'customer_kmeans.pkl')
        _MODELS['cust_features'] = joblib.load(BASE_DIR / 'customer_feature_names.pkl')
        with open(BASE_DIR / 'api_defaults.json', 'r') as f:
            _MODELS['defaults'] = json.load(f)
    return _MODELS

class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self._send_cors_headers()
        self.end_headers()
        res = {"status": "ok", "message": "ML Inference API is active on Vercel"}
        self.wfile.write(json.dumps(res).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            payload = json.loads(body.decode('utf-8'))
            action = payload.get('action')
            models = get_models()

            if action == 'predict_car':
                data = payload.get('data', {})
                input_df = pd.DataFrame([{
                    'Present_Price': float(data.get('present_price', 9.54)),
                    'Kms_Driven': int(data.get('kms_driven', 35000)),
                    'Fuel_Type': str(data.get('fuel_type', 'Petrol')),
                    'Seller_Type': str(data.get('seller_type', 'Dealer')),
                    'Transmission': str(data.get('transmission', 'Manual')),
                    'Owner': int(data.get('owner', 0)),
                    'Car_Age': int(data.get('car_age', 8)),
                    'Brand': str(data.get('brand', 'city'))
                }])
                pred = float(models['reg'].predict(input_df)[0])
                pred_bounded = max(0.1, pred)
                mae = 0.62
                response_data = {
                    'predicted_price': round(pred_bounded, 2),
                    'lower_bound': round(max(0.05, pred_bounded - mae), 2),
                    'upper_bound': round(pred_bounded + mae, 2),
                    'retention_rate': round(min(100.0, (pred_bounded / max(0.01, float(data.get('present_price', 9.54)))) * 100), 1)
                }

            elif action == 'predict_loan':
                data = payload.get('data', {})
                sample_row = dict(models['defaults']['loan_defaults'])
                
                disbursed = float(data.get('disbursed_amount', 55000))
                asset_cost = float(data.get('asset_cost', 75000))
                ltv = (disbursed / asset_cost * 100) if asset_cost > 0 else 75.0

                sample_row['DISBURSED_AMOUNT'] = disbursed
                sample_row['ASSET_COST'] = asset_cost
                sample_row['LTV'] = ltv
                sample_row['EMPLOYMENT_TYPE'] = str(data.get('employment_type', 'Salaried'))
                sample_row['PERFORM_CNS_SCORE'] = float(data.get('cns_score', 650))
                sample_row['PERFORM_CNS_SCORE_DESCRIPTION'] = str(data.get('cns_desc', 'No Bureau History Available'))
                sample_row['PRI_SANCTIONED_AMOUNT'] = float(data.get('pri_sanctioned', 50000))
                sample_row['PRI_DISBURSED_AMOUNT'] = float(data.get('pri_disbursed', 50000))
                sample_row['AADHAR_FLAG'] = int(data.get('aadhar_flag', 1))
                sample_row['PAN_FLAG'] = int(data.get('pan_flag', 0))
                sample_row['VOTERID_FLAG'] = int(data.get('voter_flag', 0))
                sample_row['DRIVING_FLAG'] = int(data.get('driving_flag', 0))

                input_df = pd.DataFrame([sample_row])
                pred_class = int(models['clf'].predict(input_df)[0])
                prob = float(models['clf'].predict_proba(input_df)[0][1] * 100)

                response_data = {
                    'prediction': pred_class,
                    'is_default': pred_class == 1,
                    'probability': round(prob, 1),
                    'ltv': round(ltv, 2),
                    'status': 'HIGH RISK' if pred_class == 1 else 'LOW RISK',
                    'message': 'Warning: High risk of loan default' if pred_class == 1 else 'Low risk: Applicant is safe'
                }

            elif action == 'predict_cluster':
                data = payload.get('data', {})
                cust_sample = dict(models['defaults']['cust_defaults'])

                cust_sample['Income'] = float(data.get('income', 65000))
                cust_sample['Education'] = str(data.get('education', 'Graduation'))
                cust_sample['Marital_Status'] = str(data.get('marital_status', 'Married'))
                cust_sample['Kidhome'] = int(data.get('kidhome', 0))
                cust_sample['Teenhome'] = int(data.get('teenhome', 1))
                cust_sample['MntWines'] = float(data.get('wine', 450))
                cust_sample['MntMeatProducts'] = float(data.get('meat', 250))
                cust_sample['MntGoldProds'] = float(data.get('gold', 60))
                cust_sample['NumWebPurchases'] = int(data.get('web_purchases', 6))
                cust_sample['NumStorePurchases'] = int(data.get('store_purchases', 8))

                input_df = pd.DataFrame([cust_sample])
                proc = models['cust_prep'].transform(input_df)
                cluster_id = int(models['cust_kmeans'].predict(proc)[0])

                profiles = {
                    0: {"name": "Luxury & Connoisseur", "color": "#8b5cf6", "desc": "High income bracket with top spend on premium wines, meats, and gourmet products."},
                    1: {"name": "Digital Active Shoppers", "color": "#38bdf8", "desc": "Moderate-high income with high engagement in digital channels and web store discounts."},
                    2: {"name": "High-Net-Worth In-Store Elites", "color": "#10b981", "desc": "Highest average income ($77.5k+) with high physical store purchases and catalogue sales."},
                    3: {"name": "Budget & Economical Families", "color": "#f59e0b", "desc": "Price-sensitive segment with lowest discretionary spending and reliance on bargain deals."}
                }
                seg = profiles.get(cluster_id, {"name": f"Segment {cluster_id}", "color": "#6366f1", "desc": "Active customer profile."})
                response_data = {
                    'cluster_id': cluster_id,
                    'segment_name': seg['name'],
                    'color': seg['color'],
                    'description': seg['desc']
                }

            else:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Invalid action'}).encode('utf-8'))
                return

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({'success': True, 'data': response_data}).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode('utf-8'))
