
import unittest
import sys
import os
import json
from flask import Flask

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from api import app, load_models

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        # Ensure models are loaded
        load_models()

    def test_predict_endpoint_roi(self):
        # Sample high-value customer data
        data = {
            "gender": "Male",
            "SeniorCitizen": "No",
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 90.0,
            "TotalCharges": 1080.0
        }
        
        response = self.app.post('/predict', 
                               data=json.dumps(data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        
        print("\nAPI Response:")
        print(json.dumps(result, indent=2))
        
        # Verify ROI fields exist
        self.assertIn('business_value', result)
        bv = result['business_value']
        self.assertIn('loss_risk', bv)
        self.assertIn('expected_savings', bv)
        self.assertIn('recommendation', bv)
        
        # Verify values check out roughly
        # CLTV should be roughly Monthly * 24 = 90 * 24 = 2160
        self.assertAlmostEqual(bv['cltv'], 2160.0, delta=100)

if __name__ == '__main__':
    unittest.main()
