Features
✅ Churn Prediction Model: Built using XGBoost, optimized with hyperparameter tuning.
✅ Power BI Dashboard: Interactive insights on churn trends and key customer behaviors.
✅ Feature Engineering: Includes computed features such as Financial Stability and Loyalty Indicator.
✅ Data Preprocessing: Outlier detection, scaling, and one-hot encoding applied.
✅ SMOTE Balancing: Adjusts churn vs. non-churn ratio to 50-50%.

Dataset Overview
The dataset includes customer details and financial attributes:

CreditScore – Customer's credit score.

Age – Age of the customer.

Balance – Account balance.

EstimatedSalary – Customer’s estimated salary.

Point Earned – Loyalty points earned.

Financial Stability – Engineered feature based on salary and balance.

Credit Risk – Computed from credit score and financial behavior.

Loyalty Indicator – Measures customer engagement.

Exited (Target Variable) – 0 = Retained, 1 = Churned.

Original Class Distribution
Non-Churn (0): 79.62%

Churn (1): 20.38%

After SMOTE Balancing: 50%-50%

Model Performance
Accuracy: 99%

Precision: High precision for identifying potential churners.

Feature Importance:

Age and CreditScore are key churn predictors.

Loyalty Indicator has a significant impact on retention.

Power BI Dashboard
📊 The Power BI report includes:

Churn rate trends by age, credit score, and tenure.

Customer segmentation based on geography, gender, and account activity.

Key insights on loyalty programs and high-risk customers.
