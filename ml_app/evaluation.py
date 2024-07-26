from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_models(data):
    # Split data into features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models for regression and classification
    regression_models = [
        ('Linear Regression', LinearRegression()),
        ('Decision Tree Regressor', DecisionTreeRegressor()),
        ('Random Forest Regressor', RandomForestRegressor()),
        ('Gradient Boosting Regressor', GradientBoostingRegressor())
    ]
    
    classification_models = [
        ('Logistic Regression', LogisticRegression()),
        ('Decision Tree Classifier', DecisionTreeClassifier()),
        ('Random Forest Classifier', RandomForestClassifier()),
        ('Gradient Boosting Classifier', GradientBoostingClassifier())
    ]
    
    results = {}
    
    # Check if target is numeric for regression
    if pd.api.types.is_numeric_dtype(y):
        for name, model in regression_models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            }
    else:
        for name, model in classification_models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1 Score': f1_score(y_test, y_pred, average='weighted'),
                'ROC AUC': roc_auc_score(y_test, y_pred, average='weighted')
            }
    
    return results
