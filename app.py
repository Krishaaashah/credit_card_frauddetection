# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For geodesic distance calculation
from geopy.distance import geodesic

# For data preprocessing and model training
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

# from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SMOTEENN

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN

from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

st.title("Credit Card Fraud Detection")

st.write("""
This app processes credit card transaction data, performs feature engineering and oversampling,
trains an XGBoost classifier (with GridSearchCV), and then lets you predict whether a transaction is fraudulent.
""")

@st.cache(allow_output_mutation=True)
def load_and_process_data():
    # --------------------------
    # 1. Load dataset
    # --------------------------
    data = pd.read_csv("credit_card_transactions.csv")
    
    # --------------------------
    # 2. Remove Outliers Based on 'amt'
    # --------------------------
    outlier_threshold = 3500
    outliers = data['amt'] > outlier_threshold
    data = data[~outliers]  # Remove outliers

    # --------------------------
    # 3. Drop Unnecessary Features
    # --------------------------
    to_drop = ['Unnamed: 0', 'first', 'last', 'street', 'city', 'state', 'zip', 
               'trans_num', 'unix_time', 'merch_zipcode']
    data = data.drop(columns=to_drop)

    # --------------------------
    # 4. Feature Creation from trans_date_trans_time
    # --------------------------
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    data['trans_year'] = data['trans_date_trans_time'].dt.year
    data['trans_month'] = data['trans_date_trans_time'].dt.month
    data['trans_day'] = data['trans_date_trans_time'].dt.day
    data['trans_season'] = data['trans_date_trans_time'].dt.month % 12 // 3 + 1  # 1=Winter, 2=Spring, etc.
    data['trans_weekday'] = data['trans_date_trans_time'].dt.weekday 
    data['trans_hour'] = data['trans_date_trans_time'].dt.hour
    data['trans_minute'] = data['trans_date_trans_time'].dt.minute
    data['trans_second'] = data['trans_date_trans_time'].dt.second
    data = data.drop(columns=['trans_date_trans_time'])
    
    # --------------------------
    # 5. Calculate Cardholder Age at Transaction Time
    # --------------------------
    data['dob'] = pd.to_datetime(data['dob'])
    data['birth_year'] = data['dob'].dt.year
    data['card_holder_age'] = data['trans_year'] - data['birth_year']
    data = data.drop(columns=['dob', 'birth_year'])
    
    # --------------------------
    # 6. Calculate Geographical Distance Using Lat/Long
    # --------------------------
    def calculate_distance(row):
        point_a = (row['lat'], row['long'])
        point_b = (row['merch_lat'], row['merch_long'])
        return geodesic(point_a, point_b).kilometers 
    data['distance'] = data.apply(calculate_distance, axis=1)
    
    # --------------------------
    # 7. Encoding Categorical Features
    # --------------------------
    def encode_categorical_columns(df, columns):
        le = LabelEncoder()
        for col in columns:
            df[col] = le.fit_transform(df[col])
        return df

    cat_features = ['cc_num', 'merchant', 'category', 'gender', 'job']
    data = encode_categorical_columns(data, cat_features)
    
    return data

data = load_and_process_data()

st.write(f"Data loaded and processed. Shape: {data.shape}")
# Display first few rows
st.dataframe(data.head())

# --------------------------
# 8. Oversampling Using a SMOTE Variant (ADASYN in this case)
# --------------------------
# Split dataset into features and target (assumes 'is_fraud' is the label column: 0=Legitimate, 1=Fraud)
X = data.drop(columns=['is_fraud'])
y = data['is_fraud']

# Split into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Choose one SMOTE technique; here we use ADASYN as an example
smote = ADASYN(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

st.write("Class distribution before SMOTE:")
st.write(y_train.value_counts())
st.write("Class distribution after SMOTE:")
st.write(pd.Series(y_train_resampled).value_counts())

# --------------------------
# 9. Train an XGBoost Model with Grid Search
# --------------------------
st.write("Training the XGBoost model... This may take a moment.")

# Define an XGBoost classifier
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    scale_pos_weight=len(y_train_resampled) / sum(y_train_resampled == 1),
    random_state=42
)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 300],
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 6],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(xgb_model, param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=0)
grid_search.fit(X_train_resampled, y_train_resampled)

best_xgb = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_xgb.predict(X_test)
y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]

st.write("### Model Performance on Test Set")
st.text(classification_report(y_test, y_pred))
st.write("AUC-ROC Score:", roc_auc_score(y_test, y_pred_proba))

# --------------------------
# 10. Streamlit Prediction Interface
# --------------------------
st.write("## Predict a New Transaction")
st.write("Enter the transaction feature values as a comma-separated list in the following order:")

# Display the expected feature order for user reference
feature_order = list(X.columns)
st.write(feature_order)

# Input field for features
input_features = st.text_input("Input features (comma-separated):", "")

if st.button("Predict Transaction"):
    if input_features:
        try:
            # Convert input string to a numpy array of floats
            feature_values = np.array([float(x.strip()) for x in input_features.split(',')])
            # Check if the number of input features matches the expected count
            if feature_values.shape[0] != len(feature_order):
                st.error(f"Expected {len(feature_order)} features, but got {feature_values.shape[0]}.")
            else:
                # Reshape and predict using the trained model
                prediction = best_xgb.predict(feature_values.reshape(1, -1))
                prediction_proba = best_xgb.predict_proba(feature_values.reshape(1, -1))[:, 1]
                if prediction[0] == 0:
                    st.success(f"Predicted: Legitimate Transaction (Probability of fraud: {prediction_proba[0]:.2f})")
                else:
                    st.error(f"Predicted: Fraudulent Transaction (Probability of fraud: {prediction_proba[0]:.2f})")
        except Exception as e:
            st.error("Error processing input. Please ensure you entered valid numerical values.")
    else:
        st.error("Please enter the feature values.")
