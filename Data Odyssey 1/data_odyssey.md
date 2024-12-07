# Data Odyssey 1 - Mochammad Daffa Putra Karyudi

## Project Overview

### Background
Customer churn is a critical issue in many subscription-based businesses. Understanding and predicting churn can help organizations implement targeted interventions to retain customers and reduce revenue loss. This project analyzes churn patterns in a telecom dataset to identify the key drivers of customer churn and propose actionable strategies for retention.

### Problem Importance
1. **Revenue Optimization**: Preventing churn directly contributes to increased customer lifetime value and revenue stability.
2. **Customer Experience Improvement**: By understanding churn drivers, businesses can tailor their offerings to meet customer needs.
3. **Strategic Insights**: Provides a data-driven approach to prioritize retention initiatives effectively.

### Dataset Overview
- **Source**: Simulated telecom data containing 7,043 entries and 21 features, including customer demographics, services subscribed, and account information.
- **Target**: The `Churn` variable indicating if a customer has left the company.
- **Features**: Data includes services like internet, streaming, tech support, and customer demographics like gender and tenure.

## Business Understanding

### Problem Statements
1. What are the primary factors influencing customer churn?
2. How can we use machine learning to predict churn effectively?
3. What strategies can reduce churn based on model insights?

### Goals
1. **Churn Prediction**: Build a robust model to classify churn with high accuracy.
2. **Feature Importance**: Identify and analyze the most significant features driving churn.
3. **Actionable Insights**: Propose targeted strategies for customer retention.

## Exploratory Data Analysis (EDA)

### 1. Correlation Analysis
#### Visualization: Correlation Heatmap
- **Description**: A heatmap visualizing the Pearson correlation between numerical features.

<iframe src="./assets/correlation_heatmap1.html" width="100%" height="800px" frameborder="0"></iframe>

![Correlation Heatmap](https://via.placeholder.com/800x400.png?text=Correlation+Heatmap)



- **Insights**:
  - Strong negative correlations:
    - `Contract Type` (-0.397): Longer contracts reduce churn risk.
    - `Tenure` (-0.352): Loyalty grows over time, lowering churn probability.
  - Moderate positive correlations:
    - `Monthly Charges` (0.193): Higher charges slightly increase churn risk.

---

### 2. Feature-wise Churn Analysis

#### Visualization: Monthly Charges Distribution
- **Description**: A box plot of monthly charges grouped by churn status.

![Monthly Charges Box Plot](https://via.placeholder.com/800x400.png?text=Monthly+Charges+Box+Plot)

- **Insights**:
  - Churned customers have a higher median monthly charge of $79.65 compared to $64.43 for non-churned customers.
  - Indicates pricing sensitivity among customers at higher billing rates.

#### Visualization: Churn Rate by Contract Type
- **Description**: Bar plot displaying churn rates across different contract types.

![Churn Rate by Contract Type](https://via.placeholder.com/800x400.png?text=Churn+Rate+by+Contract+Type)

- **Insights**:
  - Month-to-month contracts have the highest churn rate (42.7%).
  - Two-year contracts exhibit the lowest churn rate (2.8%), emphasizing the importance of long-term commitments.

#### Visualization: Tenure Distribution
- **Description**: A histogram with churn status, showing tenure distribution.

![Tenure Distribution Histogram](https://via.placeholder.com/800x400.png?text=Tenure+Distribution+Histogram)

- **Insights**:
  - New customers (tenure < 10 months) have the highest churn risk.
  - Retained customers display longer tenures, with a mode of 72 months.

#### Visualization: Churn Rate by Services
- **Description**: Subplots of churn rates for each service type (e.g., `PhoneService`, `InternetService`, `TechSupport`).

![Churn Rate by Services](https://via.placeholder.com/800x400.png?text=Churn+Rate+by+Services)

- **Insights**:
  - `Fiber Optic`: Highest churn rate among internet services (41.89%).
  - `Online Security` and `Tech Support`:
    - Without these services: ~41% churn.
    - With these services: ~15% churn, showing strong retention impact.

#### Visualization: Payment Method Analysis
- **Description**: Bar plot showing churn rates by payment method.

![Payment Method Analysis](https://via.placeholder.com/800x400.png?text=Payment+Method+Analysis)

- **Insights**:
  - Electronic checks have the highest churn rate (45.29%).
  - Automatic payments via bank transfer or credit card have the lowest churn rates (~15%).

#### Visualization: Treemap of Service Usage and Churn Rate
- **Description**: A treemap showing churn rates across service categories.

![Service Usage Treemap](https://via.placeholder.com/800x400.png?text=Service+Usage+Treemap)

- **Insights**:
  - `InternetService` type is a major predictor:
    - Fiber Optic has the highest churn rate, suggesting service issues.
    - No Internet Service customers show the lowest churn rate (7.4%).
  - Bundled services like `OnlineBackup` and `DeviceProtection` moderately improve retention.

---

## Data Preparation

### Preprocessing Steps
1. **Encoding**: Label encoding for categorical variables like Contract and Internet Service.
2. **Scaling**: StandardScaler applied to numeric features for uniformity.
3. **Splitting**: Data split into training (70%), validation (15%), and test (15%) sets.

### Challenges Addressed
- Missing values in `TotalCharges` handled by mean imputation.
- Balanced dataset using stratified sampling during splits.

## Modeling

### Algorithm Used
**Deep Neural Network (DNN)**:
- **Architecture**: 5 dense layers with LeakyReLU activation, BatchNormalization, and Dropout.
- **Regularization**: L2 regularization to prevent overfitting.
- **Metrics**: Accuracy and AUC-ROC.

### Model Performance
| Metric            | DNN (Validation) |
|--------------------|------------------|
| Accuracy           | 0.846           |
| F1 Score           | 0.810           |
| AUC-ROC Score      | 0.872           |

## Evaluation

### Findings
1. **Feature Importance**: Tenure and contract type are critical churn predictors.
2. **Customer Segmentation**:
   - Senior citizens and single customers are at higher risk.
   - Families and customers with long-term contracts are more stable.

### Business Impact
- **Retention Strategies**:
  - Promote long-term contracts.
  - Upsell security and support services.
  - Introduce pricing discounts for high-risk segments.

### Limitations
1. Imbalanced dataset impacts recall for minority class.
2. Predictions depend heavily on dataset quality and feature engineering.

## Recommendations

1. **Short-Term**:
   - Incentivize customers to switch to yearly contracts.
   - Enhance service offerings for first-year customers.
2. **Long-Term**:
   - Develop loyalty programs to reward tenure.
   - Integrate churn predictors into a real-time alert system.

---

Would you like me to generate actual example charts or refine any specific visualizations? Let me know!
