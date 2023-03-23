import timm


def timm_create_model_wrapper(method, model_name, **kwargs):
    """Create a timm encoder as a replacement for solo-learn's encoders.
    Configure the encoder via the backbone.kwargs, all keys except pretrained_weights are passed to timm.create_model
    Example config:
        backbone:
            name: "timm_universal"
            kwargs:
                model_name: "resnet50"
                in_chans: 3
                ...

    Args:
        method (str): Not used. solo.methods.BaseMethod passes the method argument to all encoders,
                      this argument is only present for this purpose.
        model_name (str): name of model to instantiate

    Keyword Args:
        See the codumentation of timm.create_model for allowed kwargs
    Returns:
        A timm encoder
    """
    return timm.create_model(model_name=model_name, num_classes=0, **kwargs)
