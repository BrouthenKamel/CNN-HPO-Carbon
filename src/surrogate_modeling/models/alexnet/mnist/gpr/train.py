import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import joblib
from sklearn.linear_model import LinearRegression

# 1. Load the full dataset
df = pd.read_csv('./datasets/accus/alexnet_mnist.csv').iloc[0:2]

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

# 6. Train/test split — use full data for both since no separate test set
X_train = X_test = X.values
y_train = y_test = y.values

# 7. Fit Gaussian Process Regressor
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, random_state=42)
gpr.fit(X_train, y_train)

# 8. Predict with uncertainty
mu, sigma = gpr.predict(X_test, return_std=True)

# 9. 95% confidence interval
z = 1.96
lower = mu - z * sigma
upper = mu + z * sigma
picp = np.mean((y_test >= lower) & (y_test <= upper))
mpiw = np.mean(upper - lower)

# 10. Save model
joblib.dump(gpr, './models/alexnet/mnist/gpr/gpr_model.pkl')

joblib.dump(X.columns, './models/alexnet/mnist/gpr/training_columns_gpr_model.pkl')
# 11. Summary
print(f"Overall predicted mean : {mu.mean():.6f}")
print(f"Overall true mean      : {y_test.mean():.6f}")
print(f"95% PICP               : {picp*100:.1f}%")
print(f"95% MPIW               : {mpiw:.6f}")
