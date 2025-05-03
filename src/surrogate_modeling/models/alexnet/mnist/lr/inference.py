import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer

def predict():
    # 1. Load the saved model + feature columns
    lr, feature_columns = joblib.load('./models/alexnet/mnist/lr/lr_model.pkl')

    # 2. Load the input data for inference
    df = pd.read_csv('./datasets/accus/alexnet_mnist.csv').iloc[5:6]

    # 3. Preprocess: drop target (if exists)
    y_true = df['test_accuracy'] if 'test_accuracy' in df else None
    df.drop(columns=['train_accuracy', 'test_accuracy'], inplace=True, errors='ignore')

    # 4. Impute missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    zero_imp = SimpleImputer(strategy='constant', fill_value=0)
    df[num_cols] = zero_imp.fit_transform(df[num_cols])
    df[cat_cols] = zero_imp.fit_transform(df[cat_cols])

    # 5. One-hot encode categoricals
    X = pd.get_dummies(df, drop_first=True)

    # 6. Align columns (important for Linear Regression)
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0  # Add missing columns with zero
    X = X[feature_columns]  # Ensure order is same

    # 7. Predict
    mu = lr.predict(X.values)

    # 8. Output results
    for i, pred in enumerate(mu):
        print(f"Sample {i+1}:")
        print(f"  Predicted value : {pred:.6f}")
        if y_true is not None:
            true = y_true.iloc[i]
            print(f"  True value      : {true:.6f}")
        print()
    return mu

print(predict())