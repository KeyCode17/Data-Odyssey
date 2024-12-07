# Streamlit Core
import streamlit as st
import streamlit_nested_layout
from streamlit_option_menu import option_menu

# Libraries
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

from keras import layers, models
from plotly.subplots import make_subplots
from keras.regularizers import l2

# Local imports
from eda1 import render_eda

dirloc = os.path.dirname(os.path.abspath(__file__))

def data_odyssey_1():
    df = pd.read_csv(os.path.join(dirloc, 'Odyssey 1  Data Set - Telco Data.csv'))
    # Separate binary (Yes/No) columns from other categorical columns
    binary_columns = ['Churn', 'PhoneService', 'PaperlessBilling', 'Partner', 'Dependents']
    other_categorical = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod', 'gender', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies']

    # Create a copy of the dataframe
    df_encoded = df.copy()

    # Manual mapping for binary columns (Yes/No)
    binary_map = {'No': 0, 'Yes': 1}
    for col in binary_columns:
        df_encoded[col] = df_encoded[col].map(binary_map)

    # Use LabelEncoder for other categorical columns
    le = LabelEncoder()
    for col in other_categorical:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')
    df_encoded['MonthlyCharges'] = pd.to_numeric(df_encoded['MonthlyCharges'], errors='coerce')
    df_encoded['tenure'] = pd.to_numeric(df_encoded['tenure'], errors='coerce')

    # Convert numeric columns and handle missing values
    # Fill NaN values with mean for numeric columns without using inplace
    numeric_columns = ['TotalCharges', 'MonthlyCharges', 'tenure']
    df_encoded[numeric_columns] = df_encoded[numeric_columns].fillna(df_encoded[numeric_columns].mean())

    # Drop customerID if it exists
    if 'customerID' in df_encoded.columns:
        df_encoded = df_encoded.drop('customerID', axis=1)

        # Separate features and target
        X = df_encoded.drop('Churn', axis=1)
        y = df_encoded['Churn']

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the enhanced dataset
        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Load model
    model_loc = os.path.join(dirloc,'best_model.keras')
    tf_model = tf.keras.models.load_model(model_loc)

    st.title("Data Odyssey 1")
    option = st.selectbox(
        "Choose Section",
        ("Project Domain", "Business Understanding", "Data Understanding", "Exploratory Data Analysis", "Machine Learning", "Feature Importance for Churn Prediction", "Conclusion"),
    )

    st.query_params["Section"]=option

    if st.query_params["Section"] == "Project Domain":
        st.header("Project Domain", divider="gray")
        st.write("In the telecommunications industry, churn rate (customers who discontinue their subscription) is a significant challenge that directly impacts revenue and business sustainability. This project aims to develop an accurate churn prediction model using machine learning techniques, helping telecommunications companies identify and retain high-risk customers.")

        st.header("Why This Problem Matters")
        st.markdown('''
- Acquiring new customers is far more expensive than retaining existing ones
- Churn prediction enables proactive intervention to prevent customer loss
- Provides deep insights into factors influencing customer discontinuation decisions
''')

        st.header("Reference")
        st.markdown('''
1. Burgund, D., Nikolovski, S., Galić, D., & Maravić, N. (2023). Pearson Correlation in Determination of Quality of Current Transformers. Sensors (Basel, Switzerland), 23. https://doi.org/10.3390/s23052704.
2. Tharwat, A. (2020). Classification assessment methods. Applied Computing and Informatics. https://doi.org/10.1016/J.ACI.2018.08.003.
3. Karyudi, M. D. P., & Zubair, A. (2024). Classifying School Scope Using Deep Neural Networks Based on Students’ Surrounding Living Environments. Journal of Computing Theories and Applications, 2(2), 290–306. https://doi.org/10.62411/jcta.11739.
''')

    elif st.query_params["Section"] == "Business Understanding":
        st.header("Business Understanding", divider="gray")
        st.header("Problem Statements")
        st.markdown('''
1. How can we predict customer churn in telecommunications services by mapping complex behavioral patterns?
2. What are the primary factors most significantly influencing customer service discontinuation?
3. How can we develop a machine learning approach that accurately forecasts churn probability with high precision?
''')

        st.header("Goals")
        st.markdown('''
1. Develop a comprehensive visualization and analysis framework that reveals intricate patterns of customer service interactions and potential churn risks.
2. Identify and systematically analyze the most critical factors driving customer decisions to discontinue telecommunications services.
3. Create an advanced predictive model using sophisticated machine learning techniques to enhance churn prediction accuracy and provide actionable insights.
''')
    
        st.header("Solution Statements")
        st.subheader("Visualization Approach:")
        st.markdown('''
- Create multi-dimensional interactive dashboards
- Design heat maps revealing customer behavior patterns
- Develop correlation visualizations of service usage and churn risk
''')

        st.subheader("Statistical Analysis Approach:")
        st.markdown('''
- Conduct comprehensive regression analysis
- Perform detailed feature importance ranking
- Generate statistical profiles of high-risk customer segments
''')

        st.subheader("Machine Learning Approach:")
        st.markdown('''
- Implement neural network models with advanced regularization
- Develop ensemble learning techniques
- Create real-time churn probability prediction mechanisms
''')

    elif st.query_params["Section"] == "Data Understanding":
        st.header("Data Understanding", divider="gray")
        st.header("Dataset Overview")
        st.markdown('''
- Total Records: 7,043 customer entries
- Features: 20 distinct variables
- Target Variable: Churn (Customer Discontinuation)
''')
        st.header("Key Variables")
        with st.expander('Key Variables', expanded=False):
            st.markdown('''
1. Target Variable
    - `Churn`: Customer service discontinuation (Yes/No)
    ''')
            st.markdown('''
2. Subscription Metrics
    - `tenure`: Customer subscription duration
    - `Contract`: Contract type 
      * Month-to-month
      * One year
      * Two year
    - `MonthlyCharges`: Monthly billing amount
    - `TotalCharges`: Cumulative charges during subscription
    ''')
            st.markdown('''
3. Service Characteristics
    - `InternetService`: Internet service type
    - `PhoneService`: Phone service availability
    - `MultipleLines`: Multiple phone line usage
    ''')
            st.markdown('''
4. Additional Services
    - `OnlineSecurity`: Online security service
    - `OnlineBackup`: Online backup service
    - `DeviceProtection`: Device protection service
    - `TechSupport`: Technical support service
    - `StreamingTV`: Streaming TV service
    - `StreamingMovies`: Streaming movies service
    ''')
            st.markdown('''
5. Demographic Variables
    - `gender`: Customer gender
    - `SeniorCitizen`: Senior citizen status (0/1)
    - `Partner`: Relationship status (Yes/No)
    - `Dependents`: Family composition (Yes/No)
    ''')
            st.markdown('''
6. Billing Characteristics
    - `PaymentMethod`: Payment channel selection
    - `PaperlessBilling`: Billing method preference
''')

        st.dataframe(df)

    elif st.query_params["Section"] == "Exploratory Data Analysis":
            # Perform EDA
            render_eda(df,df_encoded)
    
    elif st.query_params["Section"] == "Machine Learning":
        st.header("Machine Learning", divider="gray")
        with st.expander("TensorFlow Layers", expanded=False):
            st.markdown('''
<div align="center">

| Layer Type | Output Shape | Parameters | Activation | Note |
|------------|--------------|------------|------------|------|
| Input | (None, X_train.shape[1]) | X_train.shape[1] input features | - | Initial layer with input features |
| Dense | (None, 512) | 512 neurons | Linear | First dense block with L2 regularization |
| LeakyReLU | (None, 512) | - | LeakyReLU (α=0.02) | Adds non-linearity with small negative slope |
| BatchNormalization | (None, 512) | 2048 | - | Normalize layer activations |
| Dropout | (None, 512) | Rate: 0.3 | - | Aggressive regularization to prevent overfitting |
| Dense | (None, 256) | 256 neurons | Linear | Second dense block with L2 regularization |
| LeakyReLU | (None, 256) | - | LeakyReLU (α=0.02) | Adds non-linearity with small negative slope |
| BatchNormalization | (None, 256) | 1024 | - | Normalize layer activations |
| Dropout | (None, 256) | Rate: 0.25 | - | Regularization to enhance generalization |
| Dense | (None, 128) | 128 neurons | Linear | Third dense block with L2 regularization |
| LeakyReLU | (None, 128) | - | LeakyReLU (α=0.02) | Adds non-linearity with small negative slope |
| BatchNormalization | (None, 128) | 512 | - | Normalize layer activations |
| Dropout | (None, 128) | Rate: 0.2 | - | Continued regularization |
| Dense | (None, 64) | 64 neurons | Linear | Fourth dense block with L2 regularization |
| LeakyReLU | (None, 64) | - | LeakyReLU (α=0.02) | Adds non-linearity with small negative slope |
| BatchNormalization | (None, 64) | 256 | - | Normalize layer activations |
| Dropout | (None, 64) | Rate: 0.15 | - | Reduced dropout rate |
| Dense | (None, 32) | 32 neurons | Linear | Fifth dense block with L2 regularization |
| LeakyReLU | (None, 32) | - | LeakyReLU (α=0.02) | Adds non-linearity with small negative slope |
| BatchNormalization | (None, 32) | 128 | - | Normalize layer activations |
| Dropout | (None, 32) | Rate: 0.1 | - | Minimal dropout for final hidden layer |
| Dense | (None, 1) | 1 neuron | Sigmoid | Binary classification output |

</div>
''', unsafe_allow_html=True)
# BatchNormalization
# Beta (learnable shift)  
# Gamma (learnable scale)  
# Moving mean (population mean)  
# Moving variance (population variance)  

        history_loc = os.path.join(dirloc, 'historymodel.pkl')
        with open(history_loc, 'rb') as file_pi:  
            history = pickle.load(file_pi) 
        
        # Calculate and plot learning curve
        epochs = list(range(1, len(history['accuracy']) + 1))  # Convert range to list
        train_acc = history['accuracy']
        val_acc = history['val_accuracy']

        # Find best epoch
        best_epoch = np.argmax(history['val_accuracy'])

        # Create subplots for accuracy and loss
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('Model Accuracy', 'Model Loss'))

        # Add accuracy traces
        fig.add_trace(
            go.Scatter(y=history['accuracy'], name="Accuracy",
                       line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history['val_accuracy'], name="Val Accuracy",
                       line=dict(color='red')),
            row=1, col=1
        )

        # Add loss traces
        fig.add_trace(
            go.Scatter(y=history['loss'], name="Loss",
                       line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name="Val Loss",
                       line=dict(color='red')),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(height=500, width=1000)
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)

        with st.expander("Training Metrics", expanded=False):
            st.plotly_chart(fig, use_container_width=False)

        # Generate predictions  
        y_pred = tf_model.predict(X_val).flatten()  
        y_pred_classes = (y_pred > 0.5).astype(int)  

        def generate_classification_report_markdown(y_val, y_pred_classes):  
            # Get classification report  
            report = classification_report(y_val, y_pred_classes, output_dict=True)  
                
            # Prepare markdown table  
            markdown_report = """  
### Classification Report  

<div align="center"> 

| Metric | Precision | Recall | F1-Score | Support |  
|--------|-----------|--------|----------|----------|  
"""   
                
            # First section: Class-specific metrics  
            class_metrics = [  
                ('Negative Class (0)', '0'),  
                ('Positive Class (1)', '1')  
            ]  
                
            for label, key in class_metrics:  
                row = report[key]  
                markdown_report += f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1-score']:.3f} | {int(row['support'])} |\n"  
            
            # Add a blank row separator  
            markdown_report += "| | | | | |\n"  
                
            # Second section: Overall metrics  
            overall_metrics = [  
                ('Accuracy', 'accuracy'),  
                ('Macro Average', 'macro avg'),  
                ('Weighted Average', 'weighted avg')  
            ]  
                
            for label, key in overall_metrics:  
                if key == 'accuracy':  
                    # For accuracy, use the total support  
                    total_support = sum(report.get(cls, {}).get('support', 0) for cls in ['0', '1'])  
                    markdown_report += f"| {label} | - | - | {report[key]:.3f} | {int(total_support)} |\n"  
                else:  
                    row = report[key]  
                    markdown_report += f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1-score']:.3f} | {int(row['support'])} |\n"  
            markdown_report += "\n</div>" 
            return markdown_report  
        
        # Usage in Streamlit  
        try:  
            st.markdown(generate_classification_report_markdown(y_val, y_pred_classes), unsafe_allow_html=True)  
        except Exception as e:  
            st.warning(f"Error generating detailed report: {e}")  
            st.markdown(f"```\n{classification_report(y_val, y_pred_classes)}\n```")
        
        with st.expander("Detailed Report"):
            st.markdown('''
##### Performance by Class

- Negative Class (0)
    - **Precision**: 0.857 (85.7%)
      - When the model predicts a negative instance, it is correct 85.7% of the time
    - **Recall**: 0.907 (90.7%)
      - The model correctly identifies 90.7% of all actual negative instances
    - **F1-Score**: 0.881
      - Indicates a very strong performance for the negative class
      - Excellent balance between precision and recall

- Positive Class (1)
    - **Precision**: 0.690 (69.0%)
      - When the model predicts a positive instance, it is correct 69.5% of the time
    - **Recall**: 0.576 (57.6%)
      - The model correctly identifies only 57.6% of all actual positive instances
    - **F1-Score**: 0.627
      - Significantly lower performance compared to the negative class
      - Suggests notable challenges in identifying positive instances

##### Overall Performance Metrics

- Accuracy
    - **Value**: 0.820 (82.0%)
    - Might be misleading due to class imbalance
    - High accuracy is largely driven by the better performance on the majority (negative) class

- Macro Average
    - **Precision**: 0.773 (77.3%)
      - Simple average of class precisions
    - **Recall**: 0.741 (74.1%)
      - Simple average of class recalls
    - **F1-Score**: 0.754
      - Gives equal weight to both classes
      - Reflects the performance disparity between classes

- Weighted Average
    - **Precision**: 0.813 (81.3%)
    - **Recall**: 0.820 (82.0%)
    - **F1-Score**: 0.815
    - Weighted by the number of instances in each class
    - Closer to the negative class performance due to class imbalance
''')

            st.subheader("Reference")
            st.markdown('''Tharwat, A. (2020). Classification assessment methods. Applied Computing and Informatics. https://doi.org/10.1016/J.ACI.2018.08.003.''')

        # Confusion Matrix  
        cm = confusion_matrix(y_val, y_pred_classes)  

        # Confusion Matrix Heatmap  
        confusion_matrix_fig = go.Figure(data=go.Heatmap(  
            z=cm,  
            x=['Predicted No Churn', 'Predicted Churn'],  
            y=['Actual No Churn', 'Actual Churn'],  
            text=cm,  
            texttemplate="%{text}",  
            textfont={"size": 16},  
            colorscale='RdYlBu'  
        ))

        confusion_matrix_fig.update_layout(  
            title='Confusion Matrix',  
            xaxis_title='Predicted Label',  
            yaxis_title='True Label',  
            width=600,  
            height=500  
        )

        st.plotly_chart(confusion_matrix_fig, use_container_width=False)

        with st.expander("Analysis of the Confusion Matrix", expanded=False):
            st.markdown('''  
#### Analysis of the Confusion Matrix  

The confusion matrix provides a detailed breakdown of the model's performance in classifying churn (the act of customers leaving) versus no churn. Here’s a comprehensive analysis based on the provided confusion matrix:  

##### Key Metrics Derived from the Confusion Matrix  

1. **True Positives (TP)**:   
   - **160**: The number of actual churn instances correctly predicted as churn.  
   - Indicates the model's ability to identify customers who are likely to leave.  

2. **True Negatives (TN)**:   
   - **706**: The number of actual no churn instances correctly predicted as no churn.  
   - Reflects the model's effectiveness in identifying customers who are likely to stay.  

3. **False Positives (FP)**:   
   - **72**: The number of actual no churn instances incorrectly predicted as churn.  
   - Represents customers who are predicted to leave but actually do not. This can lead to unnecessary retention efforts.  

4. **False Negatives (FN)**:   
   - **118**: The number of actual churn instances incorrectly predicted as no churn.  
   - Indicates customers who are predicted to stay but actually leave. This is critical as it represents lost opportunities for proactive retention strategies.  

##### Performance Metrics  

Using the values from the confusion matrix, we can calculate several important performance metrics:  

1. **Accuracy**:  
''')  
            st.latex(r'''  
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
''') 
            st.latex(r'''
= \frac{160 + 706}{160 + 706 + 72 + 118} = \frac{866}{1,056} \approx 0.818 \text{ (81.8\%)} 
''') 

            st.markdown('''  
- Indicates that the model correctly classifies approximately 81.8% of the instances.  

2. **Precision (Positive Predictive Value)**:  
''')  
            st.latex(r'''  
\text{Precision} = \frac{TP}{TP + FP} = \frac{160}{160 + 72} \approx 0.690 \text{ (69.0\%)}  
''')  
            st.markdown('''  
- Indicates that when the model predicts churn, it is correct about 69.5% of the time.  

3. **Recall (Sensitivity or True Positive Rate)**:  
''')  
            st.latex(r'''  
\text{Recall} = \frac{TP}{TP + FN} = \frac{160}{160 + 118} \approx 0.576 \text{ (57.6\%)}  
''')  
            st.markdown('''  
- Indicates that the model correctly identifies 55.0% of actual churn instances.  

4. **F1-Score**:  
''')  
            st.latex(r'''  
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \approx 2 \times \frac{0.690 \times 0.576}{0.690 + 0.576} \approx 0.628 \text{ (62.8\%)}  
''')  
            st.markdown('''  
- The F1-score provides a balance between precision and recall, indicating moderate performance in identifying churn.  
''')

            st.subheader("Reference")
            st.markdown('''Tharwat, A. (2020). Classification assessment methods. Applied Computing and Informatics. https://doi.org/10.1016/J.ACI.2018.08.003.''')

        # ROC Curve  
        fpr, tpr, _ = roc_curve(y_val, y_pred)  
        roc_auc = auc(fpr, tpr)  

        roc_curve_fig = go.Figure()  

        roc_curve_fig.add_trace(  
            go.Scatter(x=fpr, y=tpr,   
                       name=f'ROC curve (AUC = {roc_auc:.3f})',  
                       line=dict(color='#2ECC71', width=2))  
        )  

        roc_curve_fig.add_trace(  
            go.Scatter(x=[0, 1], y=[0, 1],  
                       name='Random Classifier',  
                       line=dict(color='#E74C3C', width=2, dash='dash'))  
        )  

        roc_curve_fig.update_layout(  
            title='Receiver Operating Characteristic (ROC) Curve',  
            xaxis_title='False Positive Rate',  
            yaxis_title='True Positive Rate',  
            template='plotly_white',  
            width=800,  
            height=500,  
            showlegend=True  
        )  

        st.plotly_chart(roc_curve_fig, use_container_width=False)

        with st.expander("Analysis of the ROC Curve", expanded=False):
            st.markdown('''
The Receiver Operating Characteristic (ROC) curve is a graphical representation used to evaluate the performance of a binary classification model. Here’s a detailed analysis based on the provided ROC curve:

#### Key Components of the ROC Curve

1. **Axes**:
   - **X-axis (False Positive Rate)**: This represents the proportion of actual negatives that are incorrectly classified as positives. A lower false positive rate is desirable.
   - **Y-axis (True Positive Rate)**: This indicates the proportion of actual positives that are correctly identified by the model. A higher true positive rate is desirable.

2. **Curve**:
   - The green curve represents the model's performance across different threshold settings.
   - The curve starts at the origin (0,0) and ideally should rise steeply towards the top left corner (1,1), indicating high true positive rates with low false positive rates.

3. **Random Classifier Line**:
   - The red dashed line represents the performance of a random classifier, which would have a true positive rate equal to the false positive rate. This line serves as a baseline for comparison.
''')
            st.markdown('''
#### AUC Calculation Equation

The **AUC** is calculated by integrating the True Positive Rate (TPR) over the False Positive Rate (FPR) at various thresholds. It can be defined as:
''')
            st.latex(r'''  
AUC = \int_{0}^{1} TPR(FPR) \, dFPR
''')

            st.markdown('''
Where:
- **TPR** (True Positive Rate or Sensitivity) is defined as:
''')
            st.latex(r'''  
  TPR = \frac{TP}{TP + FN}
''')
            st.markdown('''
- **FPR** (False Positive Rate) is defined as:
''')
            st.latex(r'''  
  FPR = \frac{FP}{FP + TN}
''')

            st.markdown('''
#### Area Under the Curve (AUC)

- **AUC Value**: 0.851
  - The AUC (Area Under the Curve) quantifies the overall performance of the model. An AUC of 0.851 indicates that the model has a good ability to distinguish between the positive and negative classes.
  - AUC values range from 0 to 1:
    - **0.5**: No discrimination (equivalent to random guessing)
    - **0.7 - 0.8**: Acceptable discrimination
    - **0.8 - 0.9**: Excellent discrimination
    - **0.9 - 1.0**: Outstanding discrimination

#### Interpretation of the Curve

- **Performance**:
  - The ROC curve shows that the model performs significantly better than a random classifier, as evidenced by the curve being well above the red dashed line.
  - The curve's shape suggests that the model maintains a relatively high true positive rate while keeping the false positive rate low across various thresholds.

- **Threshold Selection**:
  - The point on the curve closest to the top left corner (0,1) represents the optimal balance between sensitivity (true positive rate) and specificity (1 - false positive rate). This point can be used to select an appropriate threshold for classification.

''')
            st.subheader("Reference")
            st.markdown('''Tharwat, A. (2020). Classification assessment methods. Applied Computing and Informatics. https://doi.org/10.1016/J.ACI.2018.08.003.''')

    elif st.query_params["Section"] == 'Feature Importance for Churn Prediction':

        with st.expander("Analysis", expanded=False):
            st.header("Feature Importance for Churn Prediction", divider="gray")
            st.markdown('''
### Analysis of Feature Importance for Churn Prediction

#### Overview
The provided Python code calculates feature importance for a churn prediction model using a TensorFlow model. It employs a permutation importance method, which assesses how the model's accuracy changes when the values of each feature are randomly shuffled. This approach helps identify which features significantly impact the model's performance.

#### Key Components of the Code

1. **Function Definition**:
   - The `calculate_feature_importance` function takes a model, feature matrix `X`, target variable `y`, feature names, and the number of repeats for permutation.
   - It evaluates the baseline accuracy of the model and then permutes each feature to measure the drop in accuracy, indicating the feature's importance.

2. **Feature Importance Calculation**:
   - For each feature, the code permutes its values and evaluates the model's accuracy.
   - The importance score for each feature is calculated as the difference between the baseline accuracy and the permuted accuracy.

3. **Visualization**:
   - A horizontal bar chart is created using Plotly to visualize the importance scores of each feature.
   - Features are colored green if they positively impact model accuracy and red if they negatively impact it.

#### Insights from the Chart

1. **Top Features**:
   - **MonthlyCharges** and **tenure** are the most important features, indicating that the amount charged monthly and the duration of service significantly influence customer churn.
   - **Contract** and **InternetService** also show considerable importance, suggesting that the type of contract and internet service can affect customer retention.

2. **Less Important Features**:
   - **DeviceProtection** is the least important feature, with a negative impact on model accuracy. This suggests that it may not be a significant factor in predicting churn, or its effect is counterproductive in this context.

3. **General Trends**:
   - Features related to billing and service duration (like **TotalCharges** and **PaymentMethod**) also play a role, highlighting the importance of financial aspects in customer retention strategies.
   - Features such as **gender**, **SeniorCitizen**, and **Dependents** appear to have minimal impact, indicating that demographic factors may not be as influential in this model.

#### Conclusion
The analysis reveals that financial metrics and service characteristics are critical in predicting customer churn. Businesses can leverage these insights to focus on improving customer satisfaction in areas that significantly impact retention, such as pricing strategies and contract offerings. Conversely, less emphasis may be placed on features like device protection, which do not contribute positively to the model's predictive power. 

''')

            st.subheader("Reference")
            st.markdown('''Karyudi, M. D. P., & Zubair, A. (2024). Classifying School Scope Using Deep Neural Networks Based on Students’ Surrounding Living Environments. Journal of Computing Theories and Applications, 2(2), 290–306. https://doi.org/10.62411/jcta.11739''')

        # Function to calculate feature importance
        def calculate_feature_importance(model, X, y, feature_names, n_repeats=10):
            # Convert X to numpy array if it's a DataFrame
            X_array = X.values if hasattr(X, 'values') else X

            baseline_score = model.evaluate(X_array, y, verbose=0)[1]  # Get baseline accuracy
            importance_scores = []
            print(f"Baseline accuracy: {baseline_score:.4f}")  # Add this for debugging

            # For each feature
            for i, feature in enumerate(feature_names):
                scores = []

                # Repeat n times
                for _ in range(n_repeats):
                    # Create a copy of the input data
                    X_permuted = X_array.copy()
                    # Permute the feature
                    X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                    # Calculate new score
                    new_score = model.evaluate(X_permuted, y, verbose=0)[1]
                    # Calculate importance as decrease in performance
                    importance = baseline_score - new_score
                    scores.append(importance)

                # Average importance over repeats
                mean_importance = np.mean(scores)
                importance_scores.append(mean_importance)

            return importance_scores

        # Get feature names if X is a DataFrame, otherwise use the provided feature_names
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else feature_names

        # Calculate feature importance with fewer repeats first to test
        importance_scores = calculate_feature_importance(tf_model, X_test, y_test, feature_names, n_repeats=5)

        # Create DataFrame with feature importance and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        importance_df = importance_df.sort_values('importance')

        # Determine colors based on importance
        colors = ['red' if x < 0 else 'green' for x in importance_df['importance']]

        # Create the diverging bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=importance_df['feature'],
            x=importance_df['importance'],
            orientation='h',
            marker_color=colors,
            marker_line_color='white',
            marker_line_width=0.5,
            opacity=0.7
        ))

        fig.update_layout(
            title={
                'text': 'Feature Importance for Churn Prediction (TensorFlow Model)',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20}
            },
            xaxis_title='Impact on Model Accuracy (+ means more important)',
            yaxis_title='Features',
            width=1000,
            height=800,
            xaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black',
                gridcolor='lightgrey',
                griddash='dash',
                dtick=0.05  # Adjust based on your importance values
            ),
            yaxis=dict(
                tickfont=dict(size=12)
            ),
            plot_bgcolor='white'
        )

        # Add a vertical line at x=0
        fig.add_vline(x=0, line_width=1, line_color="black")

        # Display
        st.plotly_chart(fig, use_container_width=False)


    elif st.query_params["Section"] == 'Conclusion':
        st.markdown('''
---
### Model Impact Evaluation on Business Understanding

#### 1. Response to Problem Statement 1:
**Develop a machine learning model to predict customer churn in telecommunications services with high accuracy using the available dataset.**

- **Does it address the problem statement?**  
  Yes, the TensorFlow-based neural network model successfully predicts customer churn with high accuracy. The model achieved an accuracy of 81.8% and an ROC-AUC score of 0.851, indicating excellent performance in distinguishing between churn and no-churn classes.

- **Has it achieved the expected goals?**  
  The goal of building a robust predictive model was achieved. Through techniques like effective data preprocessing, feature engineering, and the use of LeakyReLU activations, the model demonstrated strong predictive capabilities.

#### 2. Response to Problem Statement 2:
**Identify and analyze the most influential factors affecting customer churn.**

- **Does the proposed solution impact business decisions?**  
  Yes, the feature importance analysis provided critical insights. Significant factors include:
  - **MonthlyCharges**: High monthly bills are positively correlated with churn.
  - **tenure**: Longer subscription periods are associated with lower churn rates, emphasizing customer loyalty.
  - **Contract type**: Customers on month-to-month contracts showed the highest churn rates (42.7%), highlighting the stability offered by longer-term contracts.
  - **OnlineSecurity and TechSupport**: Their presence significantly reduces churn, emphasizing their role as retention tools.

#### 3. Response to Problem Statement 3:
**Implement enhancement techniques like hyperparameter tuning to improve prediction performance.**

- **Has it achieved the expected goals?**  
  Yes, techniques like regularization, dropout, and parameter tuning enhanced the model's accuracy and stability. Adjustments to layer architecture and dropout rates were particularly effective in reducing overfitting and improving generalization.

---

### Influential Features Identified from Analysis
- **Key Features**:
  - **MonthlyCharges**: Strong positive correlation with churn, indicating that pricing is a critical factor.
  - **tenure**: Negatively correlated with churn, showing that customer loyalty grows with time.
  - **Contract**: Longer contracts (one- or two-year) drastically reduce churn compared to month-to-month contracts.
  - **OnlineSecurity and TechSupport**: These services serve as significant retention tools, reducing churn rates by over 25%.

---

### Conclusion

1. **Model Performance**:  
   The TensorFlow-based neural network demonstrated robust performance in predicting churn. With metrics such as an accuracy of 81.8% and an ROC-AUC score of 0.851, the model is reliable for decision-making.

2. **Key Influencers**:  
   The analysis revealed that financial metrics (e.g., MonthlyCharges), subscription duration (e.g., tenure), and value-added services (e.g., OnlineSecurity, TechSupport) are crucial in influencing customer retention.

3. **Model Optimization**:  
   The application of advanced neural network techniques like LeakyReLU activations, batch normalization, and dropout improved the model’s accuracy and prevented overfitting.

4. **Business Implications**:  
   The insights enable telecom companies to create targeted retention strategies, such as:
   - Promoting long-term contracts.
   - Enhancing value-added services like security and tech support.
   - Offering customized plans for high-risk segments identified by the model.

5. **Recommendations for Further Development**:  
   - Include dynamic market variables and customer interaction data to refine predictions.
   - Explore ensemble neural network techniques for greater robustness.
   - Enhance interpretability with SHAP (SHapley Additive exPlanations) analysis.
   ''')