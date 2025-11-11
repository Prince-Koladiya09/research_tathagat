import timm

def get_model(model_name: str, **kwargs):
    """
    Creates a pre-trained PyTorch model using the timm library.
    
    Args:
        model_name (str): The name of the model to load (e.g., 'resnet50', 'vit_base_patch16_224').
        
    Returns:
        A PyTorch nn.Module instance.
    """
    try:
        # Create the model with pretrained weights and drop the final classifier layer
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        return model
    except Exception as e:
        raise ValueError(f"Could not create model '{model_name}' from timm. Error: {e}")

def list_models(filter_str: str = "*"):
    """Lists available models in timm, with an optional filter."""
    return timm.list_models(filter_str)