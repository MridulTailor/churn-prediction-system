
import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from business_value import BusinessValueCalculator

class TestBusinessValue(unittest.TestCase):
    def setUp(self):
        self.bv = BusinessValueCalculator(
            avg_monthly_charges=50,
            avg_tenure=24,
            retention_cost=50,
            retention_success_rate=0.5
        )

    def test_cltv_calculation(self):
        # 100 * 24 = 2400
        cltv = self.bv.calculate_cltv(100, 24)
        self.assertEqual(cltv, 2400)

    def test_roi_calculation_no_churn(self):
        # Churn prob 0. ROI should be small negative or zero depending on logic, 
        # but logic says: (0 * 0.5 * CLTV) - 50 = -50
        result = self.bv.calculate_roi(churn_prob=0.0, monthly_charges=100)
        self.assertEqual(result['expected_savings'], -50.0)
        self.assertEqual(result['recommendation'], "No Action Needed")

    def test_roi_calculation_high_churn(self):
        # Churn prob 1.0. CLTV = 100 * 24 = 2400.
        # Savings = (1.0 * 0.5 * 2400) - 50 = 1200 - 50 = 1150
        result = self.bv.calculate_roi(churn_prob=1.0, monthly_charges=100)
        self.assertEqual(result['expected_savings'], 1150.0)
        self.assertEqual(result['recommendation'], "Send Retention Offer")

if __name__ == '__main__':
    unittest.main()
