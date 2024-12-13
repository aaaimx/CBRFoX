import numpy as np
import logging
from sktime.distances import distance
from typing import Callable, Union
from tqdm import tqdm
def pearson(x, y):
    return np.corrcoef(x, y)[0][1]

def distance_sktime_interface(input_data_dictionary, metric, kwargs={}):
    result = np.array(
        [distance(input_data_dictionary["forecasted_window"][:, current_component],
                  input_data_dictionary["training_windows"][current_window, :, current_component],
                  metric, **kwargs)
         for current_window in tqdm(range(input_data_dictionary["windows_len"]), desc="Windows procesadas", position=0)
         for current_component in range(input_data_dictionary["components_len"])]
    ).reshape(-1, input_data_dictionary["components_len"])

    # Validating any NaN value. Whether there is any NaN value, the program stops
    if np.isnan(result).any():
        logging.error("The result of the metric computation is NaN.")
        raise ValueError("The computation returned NaN values, which is not valid for further calculations.")

    return result
# esta función solamente evalúa la cantidad de componentes que tiene el target, no las ventanas de training

# TODO Mejorar la estructura de la función
def compute_distance_interface(input_data_dictionary,
                               metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
                               kwargs):
    correlation_per_window = np.array([])
    # correlation_per_window = metric(input_data_dictionary, **kwargs)
    try:
        # Attempt to use sktime's distance interface
        return distance_sktime_interface(input_data_dictionary, metric, kwargs)
    except Exception as sktime_error:
        logging.warning(f"Failed with sktime metric: {sktime_error}")

    # Fallback to custom callable
    try:
        if callable(metric):
            return metric(input_data_dictionary, **kwargs)
        else:
            raise TypeError(f"Metric must be callable or sktime-compatible, got: {type(metric).__name__}")
    except Exception as custom_error:
        logging.error("Custom callable execution failed", exc_info=True)
        raise


# def distance_process(evaluate_component, target_component, metric, **kwargs):
#     if metric == "pearson":
#         return np.corrcoef(evaluate_component, target_component)[0][1]
#     else:
#         return distance(evaluate_component, target_component, metric, **kwargs)


