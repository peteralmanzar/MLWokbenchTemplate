from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
from typing import Any, Dict, List

def perform_grid_search(model: Any, param_grid: Dict[str, List[Any]], X_train: DataFrame, y_train: DataFrame, cv: int = 5, verbose: int = 2) -> GridSearchCV:
    """
    Perform grid search on the given model with the provided hyperparameters.

    Parameters:
    ----------
    model : Any
        The machine learning model to optimize.
    param_grid : dict
        Dictionary containing hyperparameters to try.
    X_train : DataFrame
        Training feature data.
    y_train : DataFrame
        Training target data.
    cv : int, optional
        Number of cross-validation folds (default is 5).
    verbose : int, optional
        Verbosity level (default is 2).

    Returns:
    -------
    GridSearchCV
        The fitted GridSearchCV object.

    Example usage:
    --------------
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np
    from pandas import DataFrame

    # Generate some example data
    X_train = DataFrame(np.random.rand(100, 4))  # Replace with actual training data
    y_train = DataFrame(np.random.randint(2, size=100))  # Replace with actual training labels

    # Define the KNeighborsClassifier model
    clf = KNeighborsClassifier()

    # Define the grid search parameters
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    # Perform grid search
    best_model = perform_grid_search(clf, param_grid, X_train, y_train)

    # Print the best parameters
    print(f"Best parameters found: {best_model.best_params_}")
    """

    grid_search = GridSearchCV(model, param_grid, refit=True, verbose=verbose, cv=cv)
    grid_search.fit(X_train, y_train)

    return grid_search