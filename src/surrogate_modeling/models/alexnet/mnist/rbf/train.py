import pandas as pd
import numpy as np
from smt.surrogate_models import RBF, SGP
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor

# ─── 1. Your two original DataFrames ───────────────────────────────────────────
# (Here we assume df1, df2 are already in scope, each of shape (10, 42),
#  sharing the same columns and containing 'test_accuracy' as the target.)
df = pd.read_csv('./models/alexnet/mnist/rbf/df_cleaned_2.csv')


# ─── 2. Concatenate them vertically ────────────────────────────────────────────
# df = pd.concat([df1, df2], axis=0, ignore_index=True)   # now (20, 42)

# ─── 3. Clean / target isolation ───────────────────────────────────────────────
df.drop(columns=['train_accuracy'], inplace=True, errors='ignore')
# df.dropna(subset=['test_accuracy'], inplace=True)

X = df.drop(columns=['test_accuracy'])
y = df['test_accuracy']

# ─── 4. Impute numeric & categorical ───────────────────────────────────────────
num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(exclude=[np.number]).columns

imp_num = SimpleImputer(strategy='constant', fill_value=0)
imp_cat = SimpleImputer(strategy='constant', fill_value='missing')

X[num_cols] = imp_num.fit_transform(X[num_cols])
X[cat_cols] = imp_cat.fit_transform(X[cat_cols])

# ─── 5. One-hot encode categoricals ────────────────────────────────────────────
X = pd.get_dummies(X, drop_first=True)

# _, feature_columns = joblib.load('./models/alexnet/mnist/rbf_surrogate.pkl')

# for col in feature_columns:
#     if col not in X.columns:
#         X[col] = 0

# # Discard any extra columns not in that list, and reorder
# X = X[feature_columns]

# pca = PCA(n_components=5)
# pca_res = pca.fit_transform(X)
# pc_cols = [f"PC{i+1}" for i in range(pca_res.shape[1])]
# X = pd.DataFrame(pca_res, columns=pc_cols)

# print(X)
# ─── 6. Train/test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# ─── 7. Convert to numpy arrays & reshape y ────────────────────────────────────
X_train_arr = X_train.values                # (n_train, n_features)
X_test_arr  = X_test.values                 # (n_test,  n_features)
y_train_arr = y_train.values # (n_train, 1)
y_test_arr  = y_test.values  # (n_test,  1)

# X_arr = X.values
# y_arr = y.values

# X_arr = X.to_numpy(dtype=np.float64)          # shape: (n_samples, n_features)
# y_arr = y.to_numpy(dtype=np.float64).reshape(-1, 1)  # shape: (n_samples, 1

X_train_arr = X_train.to_numpy(dtype=np.float64)     
X_test_arr = X_test.to_numpy(dtype=np.float64)     
y_train_arr = y_train.to_numpy(dtype=np.float64).reshape(-1, 1) 
y_test_arr = y_test.to_numpy(dtype=np.float64).reshape(-1, 1) 

# ─── 8. Build & train the RBF surrogate on TRAIN only ──────────────────────────
# sm = RBF(d0 = X_train_arr.shape[1],)
# sm.set_training_values(X_train_arr, y_train_arr)
# sm.train()

# sgp = SGP()
# sgp.set_training_values(X_train_arr, y_train_arr)
# sgp.set_inducing_inputs(X_train_arr)
# sgp.train()

dt = DecisionTreeRegressor()
dt.fit(X_train_arr, y_train_arr)

# ─── 9. Predict on both train & test ───────────────────────────────────────────
# y_pred_test = sm.predict_values(X_test_arr).ravel()
# y_pred_test = sgp.predict_values(X_test_arr).ravel()
y_pred_test = dt.predict(X_test_arr).ravel()

mse_test = mean_squared_error(y_test_arr, y_pred_test)
r2_test  = r2_score       (y_test_arr, y_pred_test)

print(f"Training MSE:  {mse_test:.6f}")
print(f"Training R²:   {r2_test:.4f}")

print("Actual vs Predicted on Test Set:")
for actual, pred in zip(y_test_arr.ravel(), y_pred_test):
    print(f"  Actual: {actual:.6f}    Predicted: {pred:.6f}")
# y_pred_test  = sm.predict_values(X_test_arr).ravel()

# ─── 10. (Optional) attach predictions back into DataFrames ────────────────────
# X_train = X_train.copy()
# X_test  = X_test.copy()
# X_train['rbf_pred'] = y_pred_train
# X_test ['rbf_pred'] = y_pred_test

# ─── 11. Save surrogate and feature ordering ────────────────────────────────────
joblib.dump(
    (dt, X.columns.tolist()),
    './models/alexnet/mnist/rbf_surrogate.pkl'
)

# print("Done. Shapes:")
# print("  X_train:", X_train.shape, "y_train:", y_train.shape)
# print("  X_test :", X_test.shape,  "y_test :", y_test.shape)
