import config
from models import dl_models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from models.dl_models import GRUD


#
def get_lstm_model(n_features, seq_len, nhidden, n_layers, dropout, **extra_params):
    return dl_models.MV_LSTM(
        n_features, seq_len, nhidden, n_layers, dropout, **extra_params
    )


def get_cnn_model(n_features, seq_len, kernel_size, fcl_size, dropout, **extra_params):
    return dl_models.onedCNN(
        n_features, seq_len, kernel_size, fcl_size, dropout, **extra_params
    )

def get_transformer_model(**params):
    num_features = params['n_features']
    d_model = params['d_model']
    nhead = params['nhead']
    num_layers = params['num_layers']
    dim_feedforward = params['dim_feedforward']
    dropout = params['dropout']
    return dl_models.BloodTestTransformer(num_features,d_model, nhead, num_layers, dim_feedforward, dropout)

def get_lstm_alt_model(**params):
    n_features = params["n_features"]
    use_static = params["use_static"]
    nhidden = params["nhidden"]
    dropout = params["dropout"]
    n_layers = params["n_layers"]
    return dl_models.Hiddenstate_LSTM(
        n_features=n_features,
        nhidden=nhidden,
        n_layers=n_layers,
        dropout=dropout,
        use_static=use_static
    )


def get_gru_model(**params):
    n_features = params["n_features"]
    use_static = params["use_static"]
    nhidden = params["nhidden"]
    dropout = params["dropout"]
    n_layers = params["n_layers"]
    return dl_models.MV_GRU(
        n_features=n_features,
        n_layers=n_layers,
        nhidden=nhidden,
        dropout=dropout,
        use_static=use_static
    )


def get_LR_model(**extra_params):
    # penalty=None, tol=None, C=None, class_weight=None, solver=None, l1_ratio=None,
    return LogisticRegression(
        # penalty=penalty,
        # tol=tol,
        #  C=C,
        #   class_weight=class_weight,
        #    solver=solver,
        #   l1_ratio=l1_ratio,
        max_iter=1000,
        **extra_params
    )


def get_RF_model(
    **params
):
    return RandomForestClassifier(
        **params
    )


def get_XGB_model(
    **params
):
    return XGBClassifier(
        **params
    )


def get_nn_model(
    **params
):
    if "n_layers" in params:
        n_layers = params['n_layers']
        hidden_layer_sizes = [params[f'N_neurons_in_{n}_layer'] for n in range(n_layers)]
        params['hidden_layer_sizes'] = hidden_layer_sizes
        for i in range(0, n_layers):
            del params[f'N_neurons_in_{i}_layer']
        del params['n_layers']
    return MLPClassifier(
        early_stopping=True,
        **params
    )


def get_SVC_model(C, kernel, degree, gamma, class_weight, **extra_params):
    return SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        class_weight=class_weight,
        probability=True,
        **extra_params
    )


def get_grud_model(**builder_kwargs):
    n_features = builder_kwargs["n_features"]
    X_mean = builder_kwargs["X_mean"]
    use_static = builder_kwargs["use_static"]
    nhidden = builder_kwargs["nhidden"]
    dropout = builder_kwargs["dropout"]
    return GRUD(
        input_size=n_features,
        X_mean=X_mean,
        nhidden=nhidden,
        dropout=dropout,
        use_static=use_static,
    )



model_dispatcher = {
    "LSTM": get_lstm_model,
    "LSTM_ALT": get_lstm_alt_model,
    "CNN": get_cnn_model,
    "LR": get_LR_model,
    "RF": get_RF_model,
    "XGB": get_XGB_model,
    "MLP": get_nn_model,
    "SVC": get_SVC_model,
    "GRU": get_gru_model,
    "GRUD": get_grud_model,
    "NN": get_nn_model,
    "Transformer": get_transformer_model
}
