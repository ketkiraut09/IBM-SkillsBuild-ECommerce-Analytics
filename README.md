# E-Commerce Customer Churn & Sales Analytics with AI

**Internship Program:** AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship  
**Author:** Ketki Raut  
**Project Name:** E-Commerce Customer Churn & Sales Analytics  
**Primary Code File:** `KetkiRaut_ECommerceAnalytics.ipynb`  

---

## 📌 Project Overview
This project applies data analytics, exploratory visualization, and machine learning techniques to evaluate customer transaction behaviors, analyze purchasing patterns, and predict customer churn for an e-commerce platform. By leveraging predictive modeling, businesses can proactively identify high-risk churn customers and deploy targeted retention strategies.

---

## 📊 Dataset Information
* **Dataset Name:** `ecommerce_customer_data.csv`
* **Data Source:** Synthetic dataset simulated for e-commerce transactional analysis.
* **Key Attributes:**
  * `CustomerID`: Unique identifier for each customer.
  * `Age`: Age of the customer (18–65 years).
  * `AnnualIncome`: Annual income of the customer (in USD).
  * `TotalSpend`: Cumulative dollar amount spent on the platform.
  * `PurchaseFrequency`: Total number of orders placed.
  * `DaysSinceLastPurchase`: Days elapsed since the last transaction (recency).
  * `Churn`: Target variable ($0 = \text{Active}, 1 = \text{Churned}$).

---

## 🛠️ Technologies & Libraries Used
* **Programming Language:** Python 3.x
* **Data Handling & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn
* **Environment:** Jupyter Notebook / Google Colab / VS Code

---

## 🚀 Setup & Execution Instructions

1. **Clone or Download the Repository:**
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
   ```

2. **Install Required Dependencies:**
   Ensure Python is installed, then install all required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook:**
   Open Jupyter Notebook or VS Code and execute all cells in sequence:
   ```bash
   jupyter notebook KetkiRaut_ECommerceAnalytics.ipynb
   ```

---

## 💡 Key Findings & Recommendations
* **Primary Driver of Churn:** The number of days since the last purchase (`DaysSinceLastPurchase`) is the single strongest predictor of customer churn.
* **Inactivity Threshold:** Customers remaining inactive beyond 90 days show a sharp rise in churn probability.
* **Actionable Retention Strategy:** Trigger automated re-engagement offers and personalized discounts before customers cross the 90-day inactivity mark.

---

## 📁 Repository Structure
```
├── KetkiRaut_ECommerceAnalytics.ipynb  # Complete Python code and analysis
├── requirements.txt                    # Project dependencies
├── KetkiRaut_ProjectReport.docx        # Detailed project documentation
└── README.md                           # Project overview & instructions
```
