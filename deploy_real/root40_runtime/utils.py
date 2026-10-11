import logging

def initialize_overwatch(name):
    return logging.getLogger(name)

def count_parameters(model, trainable: bool=False):
    """Return total and trainable parameter counts of a model."""
    total_params = sum((p.numel() for p in model.parameters()))
    if trainable:
        trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
        return (total_params, trainable_params)
    else:
        return total_params
