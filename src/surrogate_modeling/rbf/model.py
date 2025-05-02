import pandas as pd
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class GPRegressorSurrogate:
    def __init__(self, C_value=5.0, length_scale=0.5, alpha=1e-3, normalize_y=True, n_restarts_optimizer=200, pca_components=0.96, pretrained_model_path=None):
        self.C_value = C_value
        self.length_scale = length_scale
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self.pca_components = pca_components

        self.scaler = None
        self.pca = None
        self.model = None
        
        if pretrained_model_path:
            self.load_model(pretrained_model_path)

    def load_dataset(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def process_dataset(self, df: pd.DataFrame):
        df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns)

        X = df.drop(columns=['train_accuracy', 'test_accuracy'])
        y = df['test_accuracy']

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.pca = PCA(n_components=self.pca_components)
        X_pca = self.pca.fit_transform(X_scaled)

        return X_pca, y

    def train(self, X, y):
        kernel = C(self.C_value, (1e-3, 1e3)) * RBF(length_scale=self.length_scale, length_scale_bounds=(1e-3, 1e3))

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer
        )

        self.model.fit(X, y)

        print(f"Surrogate Model trained successfully! Kernel: {self.model.kernel_}")

    def predict(self, x):
        x_scaled = self.scaler.transform(x)
        x_pca = self.pca.transform(x_scaled)

        return self.model.predict(x_pca)

    def save_model(self, model_path):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca
        }, model_path)
        print("Model saved successfully!")

    def load_model(self, model_path):
        loaded = joblib.load(model_path)
        self.model = loaded['model']
        self.scaler = loaded['scaler']
        self.pca = loaded['pca']
        print("Model loaded successfully!")

if __name__ == "__main__":
    surrogate = GPRegressorSurrogate()
    df = surrogate.load_dataset('dataset/surrogate.csv')
    X, y = surrogate.process_dataset(df)
    surrogate.train(X, y)
    
    model_path = 'src/surrogate_modeling/rbf/models/gpr.pkl'
    
    surrogate.save_model(model_path)

    surrogate.load_model(model_path)