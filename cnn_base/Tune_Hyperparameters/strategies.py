from ..Models import CNN
from ..Models import Transformers

"""
Two examples are given for search spaces and strategies
You can make your own here
"""


def A() -> tuple :
    """
    returns : tuple(a_function_defining_the_search_space, a_function_defining_strategy_for_tuning)
    """
    def search_space_A(hp):
        """
        Search space for a typical CNN fine-tuning task.
        """
        return {
            "base_model_name": hp.Choice("base_model_name", values=["resnet50", "efficientnetb0", "mobilenetv2"]),
            "learning_rate": hp.Float("learning_rate", min_value=1e-5, max_value=1e-3, sampling="log"),
            "optimizer": hp.Choice("optimizer", values=["adam", "adamw", "sgd"]),
            "unfreeze_layers": hp.Int("unfreeze_layers", min_value=10, max_value=50, step=10)
        }

    def strategy_A(model: CNN, hp, hps: dict):
        """
        Strategy for a CNN:
        - Unfreezes the last N layers based on the hyperparameter search.
        - Compiles with the chosen optimizer and learning rate.
        """
        model.freeze_all()
        model.unfreeze_later_n(n=hps["unfreeze_layers"])
        
        model.compile(optimizer_config={
            "name": hps["optimizer"],
            "learning_rate": hps["learning_rate"]
        })
    return (search_space_A, strategy_A)

def B() -> tuple :
    """
    returns : tuple(a_function_defining_the_search_space, a_function_defining_strategy_for_tuning)
    """
    def search_space_B(hp):
        """
        Search space for a Vision Transformer fine-tuning task.
        """
        return {
            "base_model_name": "vit_base",
            "unfreeze_blocks": hp.Int("unfreeze_blocks", min_value=1, max_value=4, step=1),
            "use_llrd": hp.Boolean("use_llrd"),
            "llrd_decay_rate": hp.Float("llrd_decay_rate", min_value=0.65, max_value=0.9, sampling="linear", parent_name="use_llrd", parent_values=[True]),
            "head_lr_multiplier": hp.Float("head_lr_multiplier", min_value=1.5, max_value=5.0, sampling="linear", parent_name="use_llrd", parent_values=[True]),
            "learning_rate": hp.Float("learning_rate", min_value=1e-5, max_value=8e-5, sampling="log"),
            "weight_decay": hp.Float("weight_decay", min_value=1e-3, max_value=1e-1, sampling="log"),
        }

    def strategy_B(model: Transformers, hp, hps: dict):
        """
        Strategy for a Transformer:
        - Unfreezes the last N transformer blocks.
        - Decides whether to use Layer-wise Learning Rate Decay (LLRD).
        - If LLRD is used, tunes its decay rate.
        - If not, compiles with a standard AdamW optimizer.
        """
        model.unfreeze_last_n_blocks(n_blocks=hps["unfreeze_blocks"])
        
        if hps.get("use_llrd", False):
            model.compile_with_llrd(
                head_lr_multiplier=hps["head_lr_multiplier"],
                decay_rate=hps["llrd_decay_rate"]
            )
        else:
            model.compile(optimizer_config={
                "name": "adamw",
                "learning_rate": hps["learning_rate"],
                "params": {"weight_decay": hps["weight_decay"]}
            })
    return (search_space_B, strategy_B)