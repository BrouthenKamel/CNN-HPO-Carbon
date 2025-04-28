import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import joblib

# 1. Load the full dataset
df = pd.read_csv('./src/surrogate_modeling/datasets/accus/alexnet_mnist.csv').iloc[0:5]

# 2. Drop unwanted column and rows with no target
df.drop(columns=['train_accuracy'], inplace=True, errors='ignore')
df.dropna(subset=['test_accuracy'], inplace=True)

# 3. Split features and target
X = df.drop(columns=['test_accuracy'])
y = df['test_accuracy']

# 4. Fill missing values with 0
num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(exclude=[np.number]).columns

zero_imp = SimpleImputer(strategy='constant', fill_value=0)
X[num_cols] = zero_imp.fit_transform(X[num_cols])
X[cat_cols] = zero_imp.fit_transform(X[cat_cols])

# 5. One-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)

# 6. Train the Linear Regression model
lr = LinearRegression()
lr.fit(X, y)

# 7. Save model
joblib.dump((lr, X.columns.tolist()), './src/surrogate_modeling/datasets/accus/alexnet_mnist.csv')

# 8. Summary
print(f"Model trained. R^2 Score on training data: {lr.score(X, y):.4f}")
