import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer

def predict():
    # 1. Load the saved model
    gpr = joblib.load(f'./models/alexnet/mnist/gpr/gpr_model.pkl')
    training_columns = joblib.load('./models/alexnet/mnist/gpr/training_columns_gpr_model.pkl')

    # 2. Load the input data for inference
    df = pd.read_csv('./datasets/accus/alexnet_mnist.csv').iloc[2:3]

    # 3. Preprocess: drop target (if exists), and keep a copy if you want to compare later
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

    for col in training_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[training_columns]
    # Optional: Align with training model columns (if needed across datasets)
    # In case new input has different feature set
    # training_columns = gpr.X_train_.shape[1]  # Not directly accessible, so save columns during training if needed

    # 6. Predict with uncertainty
    mu, sigma = gpr.predict(X.values, return_std=True)

    # 7. Output results
    z = 1.96
    lower = mu - z * sigma
    upper = mu + z * sigma

    
    for i, (pred, std, l, u) in enumerate(zip(mu, sigma, lower, upper)):
        print(f"Sample {i+1}:")
        print(f"  Predicted value : {pred:.6f}")
        print(f"  95% CI          : [{l:.6f}, {u:.6f}]")
        if y_true is not None:
            true = y_true.iloc[i]
            in_interval = "Yes" if l <= true <= u else "No"
            print(f"  True value      : {true:.6f}")
            print(f"  In Interval     : {in_interval}")
        print()
    return mu

print(predict())