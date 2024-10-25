import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def split_data(data):
    # Split data into features (X) and target (y)
    X = data.drop(columns=['customerID', 'Churn'])
    y = data['Churn']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model

def train_random_forest(X_train, y_train):
    # Train Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    # Evaluate model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1

def hyperparameter_tuning(model, param_grid, X_train, y_train):
    # Perform hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

def main():
    # Load Data
    data = pd.read_csv('telecomchurn.csv')
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Train Logistic Regression model
    lr_model = train_logistic_regression(X_train, y_train)
    
    # Train Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    
    # Evaluate models
    lr_accuracy, lr_precision, lr_recall, lr_f1 = evaluate_model(lr_model, X_test, y_test)
    rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model(rf_model, X_test, y_test)
    
    print("Logistic Regression Model:")
    print("Accuracy:", lr_accuracy)
    print("Precision:", lr_precision)
    print("Recall:", lr_recall)
    print("F1-score:", lr_f1)
    
    print("\nRandom Forest Model:")
    print("Accuracy:", rf_accuracy)
    print("Precision:", rf_precision)
    print("Recall:", rf_recall)
    print("F1-score:", rf_f1)

if __name__ == "__main__":
    main()
