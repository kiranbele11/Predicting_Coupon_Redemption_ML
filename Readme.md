# 🧠 Coupon Redemption Prediction Project

Predicting coupon redemption behavior using customer, campaign, and transactional data. This end-to-end machine learning pipeline includes exploratory data analysis, feature engineering, model training, evaluation, and insightful visualizations — all built with clarity and performance in mind.

![Model Performance](https://img.shields.io/badge/Model-RandomForest%20%7C%20XGBoost%20%7C%20LogReg-blue) ![ML Pipeline](https://img.shields.io/badge/Pipeline-Full%20ML%20Workflow-brightgreen) ![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 🚀 Project Objective

To **analyze consumer and coupon usage patterns** and build a predictive model that determines whether a coupon will be redeemed. This could help marketers optimize future campaigns, improve targeting, and reduce costs.

---

## 📁 Repository Structure

├── Data_prep.ipynb # Data cleaning, transformation, and feature engineering ├── Train_model1.ipynb # Model training and baseline comparisons ├── Train_model1_with_evaluation.ipynb # Extended version with visual evaluation ├── dataset/ # Raw and cleaned data (if applicable) ├── README.md # Project overview and highlights └── requirements.txt # All dependencies 


---

## 🧱 Techniques & Tools Used

- **Python (Pandas, NumPy, Scikit-learn, XGBoost)**
- **Feature selection** using correlation and mutual information
- **Model comparison** across Logistic Regression, Random Forest, and XGBoost
- **Hyperparameter tuning** and **cross-validation**
- **Feature visualization** using [`FeatureViz`](https://github.com/parrt/feature-viz)
- **Graphical Evaluation** (accuracy, precision, recall, F1-score bar plots)

---

## 📊 Highlight: Model Evaluation Results

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 85.3%    | 84.7%     | 85.1%  | 84.9%    |
| Random Forest       | 88.9%    | 88.5%     | 88.7%  | 88.6%    |
| XGBoost             | 90.2%    | 89.8%     | 90.0%  | 89.9%    |

📈 Visualized comparison available in the notebook!

---

## 🧠 Key Learnings

- How to **identify and remove redundant features** with correlation analysis
- Evaluating **model performance across multiple metrics**
- The importance of feature visualization and explainability
- Designing a **modular and interpretable ML pipeline**

---

## 💡 Next Steps (Future Enhancements)

- Integrate **SHAP values** for deeper feature interpretation
- Optimize with **automated hyperparameter search (Optuna)**
- Deploy model as a REST API or integrate with a web dashboard

---

## 📌 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebooks step-by-step
jupyter notebook Data_prep.ipynb
jupyter notebook Train_model1.ipynb
