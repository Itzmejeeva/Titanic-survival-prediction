# Titanic-survival-prediction
Author: s. jeevan kumar
Date: 04/07/25
This project predicts passenger survival on the Titanic using machine learning. It includes data exploration, feature engineering, model training, and evaluation.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Multiple ML models compared:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Model evaluation metrics

## Requirements
- Python 3.8+
- Libraries:
  ```bash
  
---

### How to Use:
1. Create the directory structure shown above
2. Download the Titanic dataset from Kaggle and place in `data/`
3. Run the script to:
   - Train the model
   - Generate evaluation metrics
   - Save visualizations
4. The README will automatically reference the generated results

For Colab users, add this to the README:
```markdown"""
Titanic Survival Prediction ML Pipeline
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
DATA_PATH = 'data/titanic.csv'
RANDOM_STATE = 42

def load_data():
    """Load and verify dataset"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please download from Kaggle.")
    return pd.read_csv(DATA_PATH)

def preprocess(df):
    """Data cleaning and feature engineering"""
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['IsAlone'] = (df['FamilySize'] == 0).astype(int)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Dona'], 'Royal')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Convert categorical features
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked', 'Title'], drop_first=True)
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
               'FamilySize', 'IsAlone', 'Embarked_Q', 'Embarked_S',
               'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Royal']
    
    return df[features], df['Survived']

def evaluate(model, X_test, y_test):
    """Model evaluation with metrics and visualization"""
    y_pred = model.predict(X_test)
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix plot
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Died', 'Survived'],
                yticklabels=['Died', 'Survived'])
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')  # Save for README
    plt.show()

def save_results(model, accuracy):
    """Save model performance metrics"""
    os.makedirs('results', exist_ok=True)
    with open('results/metrics.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("Feature Importances:\n")
        for feat, imp in zip(X_train.columns, model.feature_importances_):
            f.write(f"{feat}: {imp:.4f}\n")

def main():
    print("=== Titanic Survival Prediction ===")
    
    try:
        # Data pipeline
        df = load_data()
        X, y = preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE)
        
        # Model training
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        
        # Evaluation
        evaluate(model, X_test, y_test)
        save_results(model, accuracy_score(y_test, model.predict(X_test)))
        
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
#Example result:
    main()              precision    recall  f1-score   support

           0       0.83      0.88      0.85       105
           1       0.81      0.74      0.77        74

    accuracy                           0.82       179
   macro avg       0.82      0.81      0.81       179
weighted avg       0.82      0.82      0.82       179
