import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from src.schema.block import CNNBlock, MLPBlock
from src.schema.model import ModelArchitecture
from src.schema.layer import ConvLayer, PoolingLayer, PoolingType, DropoutLayer, LinearLayer, ActivationLayer, ActivationType, PaddingType, AdaptivePoolingLayer
from src.schema.training import Training, OptimizerType
# from schema.model import ModelArchitecture
# from schema.block import CNNBlock, MLPBlock
# from schema.layer import ConvLayer, PoolingLayer, PoolingType, DropoutLayer, LinearLayer, ActivationLayer, ActivationType, PaddingType, AdaptivePoolingLayer
# from schema.training import Training, OptimizerType

def predict():
    # 1. Load the saved model + feature columns
    tree, feature_columns = joblib.load('./models/alexnet/mnist/dt/tree_model.pkl')

    # 2. Load the input data for inference
    df = pd.read_csv('./datasets/accus/alexnet_mnist.csv').iloc[4:6]

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

    # 6. Align columns
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0  # Fill missing feature columns
    X = X[feature_columns]  # Ensure correct order

    # 7. Predict
    mu = tree.predict(X.values)

    # 8. Output results
    for i, pred in enumerate(mu):
        print(f"Sample {i+1}:")
        print(f"  Predicted value : {pred:.6f}")
        if y_true is not None:
            true = y_true.iloc[i]
            print(f"  True value      : {true:.6f}")
        print()
    return mu

def featurize_model_architecture(ma: ModelArchitecture) -> pd.DataFrame:
    """
    Turn a ModelArchitecture into a flat dict of features,
    matching the columns in your training CSV.
    """
    features = {}

    # 1) CNN blocks
    for i, block in enumerate(ma.cnn_blocks):
        prefix = f"clb{i+1}"
        conv = block.conv_layer
        features[f"{prefix}_kernels"]     = conv.filters
        features[f"{prefix}_kernel_size"] = conv.kernel_size
        features[f"{prefix}_stride"]      = conv.stride
        # padding can be int or enum
        # activation
        features[f"{prefix}_activation"] = (
            block.activation_layer.type.value
            if block.activation_layer
            else None
        )
        # pooling
        if block.pooling_layer:
            pl = block.pooling_layer
            features[f"{prefix}_pool_type"]       = pl.type.value
            features[f"{prefix}_pool_size"]= pl.kernel_size
            features[f"{prefix}_pool_stride"]     = pl.stride
        # else:
        #     features.update({
        #         f"{prefix}_pool_type": None,
        #         f"{prefix}_pool_kernel_size": None,
        #         f"{prefix}_pool_stride": None,
        #         f"{prefix}_pool_padding": None,
        #     })

    # 2) Adaptive pooling (if any)
    # if ma.adaptive_pooling_layer:
    #     apl = ma.adaptive_pooling_layer
    #     features["adaptive_pool_type"] = apl.type.value
    #     features["adaptive_pool_output_size"] = apl.output_size
    # else:
    #     features["adaptive_pool_type"] = None
    #     features["adaptive_pool_output_size"] = None

    # 3) MLP blocks
    for j, block in enumerate(ma.mlp_blocks):
        if (j!= len(ma.mlp_blocks)-1):
            k= j + 1
        else:
            k=4    
        prefix = f"fc{k}"
        features[f"{prefix}_dropout"] = (
            block.dropout_layer.rate if block.dropout_layer else None
        )
        features[f"{prefix}_neurons"] = (
            block.linear_layer.neurons if block.linear_layer else 0
        )
        features[f"{prefix}_activation"] = (
            block.activation_layer.type.value if block.activation_layer else None
        )

    # 4) Training hyperparameters
    tr = ma.training
    features.update({
        "epochs":       tr.epochs,
        "batch_size":   tr.batch_size,
        "learning_rate":tr.learning_rate,
        "optimizer":    tr.optimizer.value,
    })

    # Return as single‐row DataFrame
    return pd.DataFrame([features])

def predict_from_config(ma: ModelArchitecture, model_path="./src/surrogate_modeling/models/alexnet/mnist/dt/tree_model.pkl"):
    # 1. load model + known feature columns
    tree, feature_columns = joblib.load(model_path)

    # 2. featurize
    df = featurize_model_architecture(ma)

    # 3. impute
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    zero_imp = SimpleImputer(strategy="constant", fill_value=0)
    df[num_cols] = zero_imp.fit_transform(df[num_cols])
    df[cat_cols] = zero_imp.fit_transform(df[cat_cols])

    # 4. one‐hot encode
    X = pd.get_dummies(df, drop_first=True)

    # 5. align to training columns
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    # 6. predict
    preds = tree.predict(X.values)
    return preds[0]

# featurize_model_architecture(AlexNetArchitecture).to_csv("try.csv", index=False)
# print(predict_from_config(AlexNetArchitecture))