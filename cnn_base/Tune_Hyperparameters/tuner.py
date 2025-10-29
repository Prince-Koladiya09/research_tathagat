import pandas as pd
import keras_tuner as kt
import keras
import os

from ..Models import get_model
from ..loggers import Logger
from ..configs.base_config import STORAGE_DIR

def save_tuner_results(tuner: kt.Tuner, project_name: str):
    logger = Logger(name=f"TunerSaver-{project_name}")
    
    results_dir = os.path.join(STORAGE_DIR, "tuning_results")
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f"{project_name}_results.csv")
    
    logger.info(f"Saving tuning results to {filepath}...")
    
    trials_data = []
    for trial in tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials)):
        trial_dict = {
            "trial_id": trial.trial_id,
            "status": trial.status,
            "score": trial.score
        }
        trial_dict.update(trial.hyperparameters.values)
        trials_data.append(trial_dict)
        
    if not trials_data:
        logger.warning("No trials found to save.")
        return

    results_df = pd.DataFrame(trials_data)
    results_df.to_csv(filepath, index=False)
    logger.info("Tuning results saved successfully.")


def tune(strategy_function, search_space_fn, train_data, validation_data, project_name):
    """
    A generic function to run a KerasTuner search.

    Args:
        strategy_function (function) : A function that takes a Model instance and an hp object
                                      and executes the fine-tuning steps.
        search_space_function (function) : A function that takes an hp object and defines the
                                    hyperparameter search space.
        You can find examples for both types in Tune_Hyperparameters/strategies.py

        train_data: Training data (e.g., (X_train, y_train)).
        val_data: Validation data (e.g., (X_val, y_val)).
        project_name (str): Name for the tuning project directory.
    """
    logger = Logger(name=f"KerasTuner-{project_name}")

    def build_model(hp):
        hyperparameters = search_space_fn(hp)
        base_model_name = hyperparameters.pop("base_model_name")
        
        model_wrapper = get_model(base_model_name)
        strategy_function(model_wrapper, hp, hyperparameters)
        
        if model_wrapper.model is None or not model_wrapper.model.compiled_loss:
            error_msg = "The model strategy function did not build and compile the model."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        return model_wrapper.model

    tuner = kt.Hyperband(
        hypermodel=build_model,
        objective="val_accuracy",
        max_epochs=20,
        factor=3,
        directory="tuning_results",
        project_name=project_name,
        overwrite=True
    )

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    logger.info(f"--- Starting hyperparameter search for '{project_name}' ---")
    tuner.search(train_data, validation_data=validation_data, callbacks=[stop_early])
    logger.info(f"--- Search complete for '{project_name}' ---")
    
    try:
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info("Best Hyperparameters found:")
        for key, value in best_hps.values.items():
            logger.info(f"  - {key}: {value}")
    except IndexError:
        logger.warning("Could not retrieve best hyperparameters. The search may have failed.")
    
    save_tuner_results(tuner, project_name)
            
    return tuner