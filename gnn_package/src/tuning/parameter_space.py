# gnn_package/src/tuning/parameter_space.py

from typing import Dict, Any, Callable, Optional
import optuna


def get_default_param_space() -> Dict[str, Callable[[optuna.trial.Trial], Any]]:
    """
    Define the default hyperparameter search space.
    Focuses on most impactful parameters for initial tuning.

    Returns:
    --------
    Dict[str, Callable]
        Dictionary mapping parameter names to trial suggest functions
    """
    param_space = {
        # Model architecture parameters
        "model.hidden_dim": lambda trial: trial.suggest_categorical(
            "model.hidden_dim", [32, 64, 128, 256]
        ),
        "model.num_layers": lambda trial: trial.suggest_int("model.num_layers", 1, 3),
        "model.num_gc_layers": lambda trial: trial.suggest_int(
            "model.num_gc_layers", 1, 3
        ),
        "model.dropout": lambda trial: trial.suggest_float("model.dropout", 0.1, 0.5),
        # Training parameters
        "training.learning_rate": lambda trial: trial.suggest_float(
            "training.learning_rate", 1e-4, 1e-2, log=True
        ),
        "training.weight_decay": lambda trial: trial.suggest_float(
            "training.weight_decay", 1e-6, 1e-3, log=True
        ),
    }

    return param_space


def get_focused_param_space(
    previous_best_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Callable[[optuna.trial.Trial], Any]]:
    """
    Define a more focused hyperparameter search space,
    optionally centered around previous best parameters.

    Parameters:
    -----------
    previous_best_params : Dict[str, Any], optional
        Best parameters from a previous tuning run

    Returns:
    --------
    Dict[str, Callable]
        Dictionary mapping parameter names to trial suggest functions
    """
    if previous_best_params is None:
        return get_default_param_space()

    # Create a more focused search around previous best values
    param_space = {}

    # Focus hidden_dim search
    if "model.hidden_dim" in previous_best_params:
        best_hidden = previous_best_params["model.hidden_dim"]
        # Get neighboring values, ensuring we stay within reasonable ranges
        hidden_options = [
            max(16, best_hidden // 2),
            best_hidden,
            min(512, best_hidden * 2),
        ]
        # Remove duplicates and sort
        hidden_options = sorted(list(set(hidden_options)))
        param_space["model.hidden_dim"] = lambda trial: trial.suggest_categorical(
            "model.hidden_dim", hidden_options
        )
    else:
        param_space["model.hidden_dim"] = lambda trial: trial.suggest_categorical(
            "model.hidden_dim", [64, 128, 256]
        )

    # Focus num_layers search
    if "model.num_layers" in previous_best_params:
        best_layers = previous_best_params["model.num_layers"]
        param_space["model.num_layers"] = lambda trial: trial.suggest_int(
            "model.num_layers", max(1, best_layers - 1), min(4, best_layers + 1)
        )
    else:
        param_space["model.num_layers"] = lambda trial: trial.suggest_int(
            "model.num_layers", 1, 3
        )

    # Focus num_gc_layers search
    if "model.num_gc_layers" in previous_best_params:
        best_gc_layers = previous_best_params["model.num_gc_layers"]
        param_space["model.num_gc_layers"] = lambda trial: trial.suggest_int(
            "model.num_gc_layers",
            max(1, best_gc_layers - 1),
            min(4, best_gc_layers + 1),
        )
    else:
        param_space["model.num_gc_layers"] = lambda trial: trial.suggest_int(
            "model.num_gc_layers", 1, 3
        )

    # Focus dropout search
    if "model.dropout" in previous_best_params:
        best_dropout = previous_best_params["model.dropout"]
        param_space["model.dropout"] = lambda trial: trial.suggest_float(
            "model.dropout", max(0.05, best_dropout - 0.1), min(0.6, best_dropout + 0.1)
        )
    else:
        param_space["model.dropout"] = lambda trial: trial.suggest_float(
            "model.dropout", 0.1, 0.5
        )

    # Focus learning rate search
    if "training.learning_rate" in previous_best_params:
        best_lr = previous_best_params["training.learning_rate"]
        param_space["training.learning_rate"] = lambda trial: trial.suggest_float(
            "training.learning_rate", best_lr / 3, best_lr * 3, log=True
        )
    else:
        param_space["training.learning_rate"] = lambda trial: trial.suggest_float(
            "training.learning_rate", 1e-4, 1e-2, log=True
        )

    # Focus weight decay search
    if "training.weight_decay" in previous_best_params:
        best_wd = previous_best_params["training.weight_decay"]
        param_space["training.weight_decay"] = lambda trial: trial.suggest_float(
            "training.weight_decay", best_wd / 5, best_wd * 5, log=True
        )
    else:
        param_space["training.weight_decay"] = lambda trial: trial.suggest_float(
            "training.weight_decay", 1e-6, 1e-3, log=True
        )

    return param_space


def get_param_space_with_suggestions(
    trial: optuna.trial.Trial,
    param_space: Dict[str, Callable[[optuna.trial.Trial], Any]],
) -> Dict[str, Any]:
    """
    Generate parameter values from the search space for a specific trial.

    Parameters:
    -----------
    trial : optuna.trial.Trial
        Current Optuna trial object
    param_space : Dict[str, Callable]
        Parameter space definition

    Returns:
    --------
    Dict[str, Any]
        Dictionary of parameter names and suggested values
    """
    params = {}
    for param_name, suggest_func in param_space.items():
        params[param_name] = suggest_func(trial)

    return params
