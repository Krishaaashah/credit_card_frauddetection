#  Fraud Detection using Machine Learning

This project aims to develop a robust and interpretable machine learning model to detect fraudulent banking transactions using real-world features. The solution handles imbalanced data using SMOTE, leverages ensemble learning techniques, and evaluates performance with relevant metrics to ensure real-time fraud prevention.

---

##  Problem Statement

To build a machine learning-based fraud detection system that accurately classifies legitimate and fraudulent transactions using structured transaction data.

---

## Features Used

- `cc_num`: Credit card number (encoded)
- `merchant`, `category`, `job`: Encoded categorical features
- `amt`, `lat`, `long`, `merch_lat`, `merch_long`: Transaction & location details
- `city_pop`, `card_holder_age`: Demographic information
- `trans_year`, `trans_month`, `trans_day`, `trans_hour`, `trans_minute`, `trans_second`, `trans_weekday`, `trans_season`: Temporal patterns
- `distance`: Geo-distance between cardholder and merchant

---

## Workflow

- **Data Preprocessing**: Cleaned null values, encoded categorical features, and normalized numerical ones.
- **Handling Imbalanced Data**: Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset.
- **Baseline Model**: Trained Logistic Regression as the benchmark model.
- **Advanced Models**: Implemented and tuned **XGBoost**, **LightGBM**, and **CatBoost** classifiers.
- **Model Evaluation**: Used metrics like Precision, Recall, F1-Score, AUC-ROC, and Confusion Matrix.
- **Hyperparameter Tuning**: Optimized model performance using **GridSearchCV**.
- **Deployment**: Developed an **API using FastAPI**, and deployed it to the cloud via **Render**.
  

https://github.com/user-attachments/assets/069c43ff-1d20-43e2-9a72-9c01d7a3e4f8

Accepts raw transaction features and returns a fraud prediction in real-time.
- 

---

## Evaluation Metrics

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
- **AUC-ROC**: Measures model's ability to distinguish between classes.

---

## Tech Stack

- **Programming**: Python
- **Libraries**: Scikit-learn, XGBoost, CatBoost, LightGBM, Pandas, NumPy, Matplotlib, Imbalanced-learn
- **Model Serving**: FastAPI
- **Deployment**: Render
- **Version Control**: Git + GitHub

---

