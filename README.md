# End-to-End-Machine-Learning-Project

# **Customer Churn Prediction: End-to-End Machine Learning Project**

## **Overview**
This project demonstrates an end-to-end machine learning pipeline to predict customer churn for a telecommunications company. By identifying at-risk customers, the company can take proactive steps to improve retention and reduce revenue loss.

The pipeline covers the complete data science lifecycle, including:
- **Data Ingestion**
- **Preprocessing & Feature Engineering**
- **Exploratory Data Analysis (EDA)**
- **Model Development & Training**
- **Model Evaluation**
- **Deployment using Flask API**
- **API Testing for Prediction**

---

## **Key Objectives**
1. Predict customer churn with high accuracy using machine learning.
2. Analyze factors influencing churn and provide actionable insights.
3. Build a scalable and reusable pipeline for deployment.
4. Demonstrate model usability through API endpoints.

---

## **Table of Contents**
1. [Data Ingestion](#data-ingestion)
2. [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Model Development & Training](#model-development--training)
5. [Prediction and API Deployment](#prediction-and-api-deployment)
6. [API Testing](#api-testing)
7. [Results and Insights](#results-and-insights)

---

## **1. Data Ingestion**
- **Objective**: Load and validate the raw dataset to ensure it is ready for analysis.
- **Steps**:
  - Dataset loaded from `/data/Telco_customer_churn.csv` (7043 rows, 33 columns).
  - Checked for missing values and duplicates.
  - Saved validated data to `/data/processed` for further steps.

**Technologies**: `pandas`, `os`

---

## **2. Preprocessing & Feature Engineering**
- **Objective**: Clean and prepare the data for machine learning.
- **Steps**:
  - Handled missing values using imputation strategies (`Don't know` for categorical features).
  - Removed irrelevant features (e.g., `Zip Code`).
  - Categorical encoding: Applied One-Hot and Frequency Encoding.
  - Engineered new features like:
    - **Average Monthly Spend**
    - **Churn Risk Category**
    - **Lifetime Revenue**
- **Outcome**: A clean and enriched dataset ready for training.

**Technologies**: `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## **3. Exploratory Data Analysis (EDA)**
- **Objective**: Uncover patterns and relationships in the dataset.
- **Steps**:
  - Performed univariate and bivariate analyses to study distributions and churn trends.
  - Analyzed correlations using a heatmap and visualizations.
  - Key findings:
    - High churn rates in customers with low tenure.
    - Significant churn among customers using month-to-month contracts.

**Technologies**: `matplotlib`, `seaborn`

---

## **4. Model Development & Training**
- **Objective**: Develop machine learning models to predict churn.
- **Steps**:
  - Baseline model: Logistic Regression.
  - Addressed class imbalance using SMOTE (Synthetic Minority Oversampling Technique).
  - Trained advanced models:
    - Random Forest
    - Gradient Boosting
    - XGBoost
  - Hyperparameter tuning to optimize performance.
  - Evaluated models using:
    - **Precision, Recall, F1-Score**
    - **ROC AUC**
  - Gradient Boosting was the best-performing model with:
    - **ROC AUC**: 0.97
    - **Recall for churned customers**: 90%

**Technologies**: `scikit-learn`, `imbalanced-learn`, `xgboost`

---

## **5. Prediction and API Deployment**
- **Objective**: Deploy the trained model to predict churn on new data.
- **Steps**:
  - Saved the optimized model (`optimized_gradient_boosting_model.pkl`).
  - Built a Flask API to serve predictions.
  - API functionality:
    - Accepts customer features as JSON input.
    - Returns churn predictions and probabilities.

**Technologies**: `Flask`, `joblib`

---

## **6. API Testing**
- **Objective**: Validate the API's robustness and reliability.
- **Steps**:
  - Tested API with valid and invalid inputs.
  - Simulated edge cases:
    - Missing fields
    - Incorrect data types
    - Out-of-range values
  - Ensured proper error handling and response consistency.
- **Outcome**: API is robust and ready for deployment.

**Technologies**: `requests`, `pytest`

---

## **7. Results and Insights**
- **Key Metrics**:
  - **Precision**: 0.79 (churned customers)
  - **Recall**: 0.90 (churned customers)
  - **Accuracy**: 90%
  - **ROC AUC**: 0.97
- **Business Implications**:
  - Identified high-risk customer segments for retention efforts.
  - Month-to-month contract customers and those with low tenure are most likely to churn.
  - Retention strategies:
    - Offer discounts to high-risk customers.
    - Target new contract offers to customers with low tenure.

---

## **Project Workflow**
1. Data Loading and Preprocessing
2. Exploratory Data Analysis
3. Feature Engineering
4. Model Development and Training
5. Model Evaluation and Deployment
6. API Testing

---

## **Technologies Used**
- **Programming Languages**: Python
- **Libraries**: 
  - Data Handling: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `xgboost`, `imbalanced-learn`
  - Deployment: `Flask`, `joblib`
- **Tools**: Jupyter Notebook

---

## **Usage**
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebooks in order:
   - `data_ingestion.ipynb`
   - `preprocessing.ipynb`
   - `eda.ipynb`
   - `model_training.ipynb`
   - `predict.ipynb`
   - `api_testing.ipynb`
4. Start the Flask server:
   ```bash
   python app.py
   ```
5. Use the API to make predictions:
   - POST request to `/predict` with a JSON payload containing customer features.

---

## **Next Steps**
1. Deploy the Flask API on a cloud platform (e.g., AWS, GCP).
2. Create a dashboard for stakeholders to interact with predictions.
3. Integrate feedback loops for continuous model improvement.

---
"""


