import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,precision_score,recall_score,f1_score,roc_curve,auc, classification_report,ConfusionMatrixDisplay
# Read data
def readData():
    data = pd.read_csv('data.csv')
    data = data.drop_duplicates()  # keep changes
    return data

# Check and convert inaccurate data types
def checkDataAccuracy(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            converted = pd.to_numeric(data[col], errors='coerce')
            valid = converted.notna().mean()
            if valid >= 0.9:
                data[col] = converted
    return data

# Handle missing values
def checkIfNull(data):
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype in ['float64', 'int64']:
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode()[0])
    return data

# Full preparation 
def dataPreparation():
    data = readData()
    data = checkDataAccuracy(data)
    data = checkIfNull(data)
    # Drop irrelevant columns if they exist
    to_drop = [col for col in ['gender', 'customerID'] if col in data.columns]
    data = data.drop(columns=to_drop)
    return data  

#encodage, 
def encodage(data):
    # Binary Yes/No columns
    yes_no_cols = ['Partner', 'Dependents','PhoneService','PaperlessBilling','Churn']
    for col in yes_no_cols:
        if col in data.columns:
            data[col] = (
                data[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({'yes': 1, 'no': 0})
                .fillna(0)
            )

    #  Categorical columns 
    cat_columns = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'InternetService', 'StreamingTV',
        'StreamingMovies', 'MultipleLines', 'Contract', 'PaymentMethod'
    ]
    cat_columns = [col for col in cat_columns if col in data.columns]

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded = pd.DataFrame(
        encoder.fit_transform(data[cat_columns]),
        columns=encoder.get_feature_names_out(cat_columns),
        index=data.index
    )
    # Combine
    data = pd.concat([data.drop(columns=cat_columns), encoded], axis=1)

    return data
def chooseXy(data):
   # Separate features and target
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    return X,y

# split Train/Test
def splitData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

#normalisation, 
def scaleData(X_train, X_test):
    need_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[need_scale] = scaler.fit_transform(X_train[need_scale])
    X_test_scaled[need_scale] = scaler.transform(X_test[need_scale])

    return X_train_scaled, X_test_scaled

    
def prepareDataset():
    data = dataPreparation()
    data = encodage(data)
    X,y=chooseXy(data)
    X_train, X_test, y_train, y_test = splitData(X, y)
    X_trainScaled,X_testScaled = scaleData(X_train, X_test)
    return X_trainScaled, X_testScaled, y_train, y_test

# entraînement des modèles
def modelTraining(model):
     X_train, X_test, y_train, y_test = prepareDataset()
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     y_scores = model.predict_proba(X_test)[:, 1]
     print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
     print(f"Precision: {precision_score(y_test, y_pred):.2f}")
     print(f"Recall:    {recall_score(y_test, y_pred):.2f}")
     print(f"F1-score:  {f1_score(y_test, y_pred):.2f}")

    # ROC Curve
     fpr, tpr, _ = roc_curve(y_test, y_scores)
     roc_auc = auc(fpr, tpr)

     plt.figure()
     plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", linewidth=2)
     plt.plot([0, 1], [0, 1], "k--") 
     plt.xlabel("False Positive Rate")
     plt.ylabel("True Positive Rate (Recall)")
     plt.title("ROC Curve")
     plt.legend()
     plt.show()


