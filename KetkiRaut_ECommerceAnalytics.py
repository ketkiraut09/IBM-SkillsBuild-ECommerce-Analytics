"""
AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship
Project Title: E-Commerce Customer Churn & Sales Analytics with AI
Author: Ketki Raut
File Name: KetkiRaut_ECommerceAnalytics.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("=" * 60)
    print(" AICTE | IBM SkillsBuild Data Analytics Internship Project")
    print(" Author: Ketki Raut")
    print(" Topic: E-Commerce Customer Churn & Sales Analytics with AI")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. Dataset Generation & Loading
    # -------------------------------------------------------------
    print("\n[Step 1] Generating and Loading Dataset...")
    np.random.seed(42)
    n_samples = 500

    data = pd.DataFrame({
        'CustomerID': range(1001, 1001 + n_samples),
        'Age': np.random.randint(18, 65, size=n_samples),
        'AnnualIncome': np.random.randint(20000, 120000, size=n_samples),
        'TotalSpend': np.random.randint(100, 5000, size=n_samples),
        'PurchaseFrequency': np.random.randint(1, 50, size=n_samples),
        'DaysSinceLastPurchase': np.random.randint(1, 365, size=n_samples),
        'Churn': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    })

    # Save local CSV copy
    data.to_csv('ecommerce_customer_data.csv', index=False)
    print("-> Dataset created successfully as 'ecommerce_customer_data.csv'")
    print("\nFirst 5 Rows of Dataset:")
    print(data.head())

    # -------------------------------------------------------------
    # 2. Exploratory Data Analysis (EDA)
    # -------------------------------------------------------------
    print("\n[Step 2] Performing Exploratory Data Analysis (EDA)...")
    print("\nDataset Info:")
    data.info()

    print("\nSummary Statistics:")
    print(data.describe())

    print("\nChecking Missing Values:")
    print(data.isnull().sum())

    # Visualizations
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(data['TotalSpend'], kde=True, color='purple')
    plt.title('Customer Total Spend Distribution')
    plt.xlabel('Total Spend ($)')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.boxplot(x='Churn', y='DaysSinceLastPurchase', data=data, palette='Set2')
    plt.title('Recency (DaysSinceLastPurchase) vs Churn')
    plt.xlabel('Churn Status (0 = Active, 1 = Churned)')
    plt.ylabel('Days Since Last Purchase')

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # 3. Model Preparation & Training
    # -------------------------------------------------------------
    print("\n[Step 3] Training Machine Learning Model (Random Forest)...")
    X = data[['Age', 'AnnualIncome', 'TotalSpend', 'PurchaseFrequency', 'DaysSinceLastPurchase']]
    y = data['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_score=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("-> Model training complete!")

    # -------------------------------------------------------------
    # 4. Evaluation & Results
    # -------------------------------------------------------------
    print("\n[Step 4] Evaluating Model Performance...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature Importance Analysis
    importances = model.feature_importances_
    features = X.columns
    feature_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    print("\nFeature Importance Breakdown:")
    print(feature_df)

    print("\n" + "=" * 60)
    print(" Execution Completed Successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
