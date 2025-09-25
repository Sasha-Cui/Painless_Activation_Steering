# util.py
def get_layer(model, layer_index):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_index]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_index]
    else:
        raise ValueError("Unrecognized model architecture for layer access.")