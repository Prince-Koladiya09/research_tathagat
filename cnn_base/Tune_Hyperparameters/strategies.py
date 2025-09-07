from keras import layers

"""
Two examples are given for search spaces and strategies
You can make your own here
"""


def A() -> tuple :
    """
    returns : tuple(a_function_defining_the_search_space, a_function_defining_strategy_for_tuning)
    """
    def search_space_A(hp) -> dict :
        return {
            "optimizer": hp.Choice("optimizer", values=["adam", "rmsprop"]),
            "learning_rate": hp.Float("learning_rate", min_value=1e-5, max_value=1e-3, sampling="log")
        }

    def strategy_A(model, hp):
        """
        Strategy A: A simple transfer learning approach.
        - Uses a fixed base model.
        - Tunes how many layers to unfreeze.
        """
        base_model_name = "ResNet50"
        num_layers_to_unfreeze = hp.Int("unfreeze_layers", min_value=10, max_value=40, step=10)

        model.get_base_model(base_model_name)
        model.freeze_all()
        model.unfreeze_later_N(num_layers_to_unfreeze)
        model.add_custom_layers() # Uses default custom layers
        model.compile()

    return (search_space_A, strategy_A)

def B() -> tuple :
    """
    returns : tuple(a_function_defining_the_search_space, a_function_defining_strategy_for_tuning)
    """
    def search_space_B(hp):
        return {
            "optimizer": "adamw", # Fixed optimizer
            "learning_rate": 1e-4, # Fixed learning rate
            "num_classes": 4 # Your config can be used here too
        }

    def strategy_B(model, hp):
        """
        Strategy B: A more complex approach.
        - Uses a powerful base model.
        - Tunes which block to cut the model at.
        - Tunes the dropout rate and number of units in a custom head.
        """
        base_model_name = "ResNet50"
        
        # Let's imagine ResNet50 has these layers to choose from
        cut_layer_name = hp.Choice("cut_at_layer", values=["conv4_block6_out", "conv3_block4_out"])
        
        # Define a tunable custom head
        hp_units = hp.Int("dense_units", min_value=128, max_value=512, step=128)
        hp_dropout = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)

        custom_head = [
            layers.GlobalAveragePooling2D(),
            layers.Dense(units=hp_units, activation="relu"),
            layers.Dropout(hp_dropout),
            layers.Dense(model.config["num_classes"], activation="softmax")
        ]

        model.get_base_model(base_model_name)
        model.cut_at_layer_and_add_custom_layers(
            layer_name=cut_layer_name,
            layers_list=custom_head
        )
        # For this strategy, we decide to train all unfrozen layers
        model.unfreeze_all()
        model.compile()

    return (search_space_B, strategy_B)