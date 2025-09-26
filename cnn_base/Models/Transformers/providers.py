from transformers import TFAutoModel


_MODEL_DICT = {
    "vit-base": "google/vit-base-patch16-224-in21k",
    "vit-large": "google/vit-large-patch16-224-in21k",
    "swin-transformer": "microsoft/swin-base-patch4-window7-224-in22k",
    "swin-v2": "microsoft/swinv2-base-patch4-window12-192-22k",
    "deit-small": "facebook/deit-small-distilled-patch16-224",
    "deit-base": "facebook/deit-base-distilled-patch16-224",
    "beit": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "convnext": "facebook/convnext-base-224-22k",
    "mobilevit": "apple/mobilevit-small",
    
    # Not implemented yet
    # "pvt" (Pyramid Vision Transformer)
    # "t2t-vit"
    # "cvt" (Convolutional Vision Transformer)
    # "vitaev2"
    # "efficientformer-l1"
    # "convmixer"
    # "poolformer"
    # "twins-svt"
    # "hrnet"
    # "bit"
    # "noisy-student"
}


def get_hf_model(model_name: str, logger):
    hf_path = _MODEL_DICT.get(model_name.lower())
    if not hf_path:
        return None
    
    try:
        logger.info(f"Loading model '{hf_path}' from Hugging Face Hub...")
        model = TFAutoModel.from_pretrained(hf_path, from_pt=True)
        return model
    except Exception as e:
        logger.error(f"Error loading model '{hf_path}' from Hugging Face Hub: {e}")
        return None