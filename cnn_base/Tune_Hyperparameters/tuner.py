import keras_tuner as kt
import tensorflow as tf
from ..Models import Model

def tune(strategy_function, search_space_fn, train_data, val_data, project_name, verbose : int = 0) :
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

    def build_model(hp) :
        """KerasTuner's model-building function."""
        
        model_config = search_space_fn(hp)

        model_wrapper = Model(update_config_kwargs=model_config)

        strategy_function(model_wrapper, hp)
        
        # The strategy must include a compile step. Return the built model.
        if model_wrapper.model is None:
            raise RuntimeError("The model strategy function did not build the model.")
        
        return model_wrapper.model

    tuner = kt.Hyperband(
        build_model,
        objective="val_accuracy",
        max_epochs=20,
        factor=3,
        directory="tuning_results",
        project_name=project_name
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    if verbose :
        print(f"--- Starting hyperparameter search for '{project_name}' ---")
    tuner.search(*train_data, validation_data=val_data, callbacks=[stop_early])

    if verbose :
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        print(f"--- Search complete for '{project_name}' ---")
        print("Best Hyperparameters:")
        # Pretty print the dictionary of best hyperparameters
        for key, value in best_hps.values.items():
            print(f"  - {key}: {value}")
        
    return tuner