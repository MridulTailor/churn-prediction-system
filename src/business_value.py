class BusinessValueCalculator:
    def __init__(self, avg_monthly_charges=70.0, avg_tenure=24, retention_cost=50.0, retention_success_rate=0.5):
        """
        Initialize with default business assumptions.
        :param avg_monthly_charges: Default monthly charges if missing
        :param avg_tenure: Default remaining tenure (months) if missing
        :param retention_cost: Cost of the intervention (e.g., discount amount)
        :param retention_success_rate: Probability that the intervention saves the customer
        """
        self.avg_monthly_charges = avg_monthly_charges
        self.avg_tenure = avg_tenure
        self.retention_cost = retention_cost
        self.retention_success_rate = retention_success_rate

    def calculate_cltv(self, monthly_charges, tenure_months=None):
        """
        Calculate Customer Lifetime Value (CLTV).
        Simple model: Monthly Charges * Remaining Tenure
        """
        if tenure_months is None:
            tenure_months = self.avg_tenure
        
        # If we assume they stay for another 2 years by default if they don't churn at this moment
        remaining_tenure = 24  
        
        return monthly_charges * remaining_tenure

    def calculate_roi(self, churn_prob, monthly_charges):
        """
        Calculate the expected ROI of intervening.
        
        Scenario A: Don't Intervene
        - Expected Value = (1 - churn_prob) * CLTV
        (We lose everything if they churn)

        Scenario B: Intervene
        - Cost = retention_cost
        - If they were going to churn (prob = churn_prob):
            - We save them with prob = retention_success_rate. Value = CLTV - Cost.
            - We fail to save them with prob = 1 - retention_success_rate. Value = -Cost.
        - If they were NOT going to churn (prob = 1 - churn_prob):
            - We wasted the money. Value = CLTV - Cost.

        Benefit of Intervention = E(Intervene) - E(Don't Intervene)
        
        Simplified logic for User Display:
        - "Loss Risk": churn_prob * CLTV
        - "Expected Savings": (churn_prob * retention_success_rate * CLTV) - retention_cost
        """
        cltv = self.calculate_cltv(monthly_charges)
        
        loss_risk = churn_prob * cltv
        
        # Expected Savings Calculation
        # Saving = (Money Saved) - Cost
        # Money Saved = (Probability they churn AND we save them) * CLTV
        expected_savings = (churn_prob * self.retention_success_rate * cltv) - self.retention_cost
        
        recommendation = "No Action Needed"
        if expected_savings > 0:
            recommendation = "Send Retention Offer"
            
        return {
            'cltv': round(cltv, 2),
            'loss_risk': round(loss_risk, 2),
            'expected_savings': round(expected_savings, 2),
            'recommendation': recommendation,
            'retention_cost': self.retention_cost
        }
