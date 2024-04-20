MODEL_REGISTRY = {}


def register_model(name):

    def _register(cls):
        if name in MODEL_REGISTRY:
            return MODEL_REGISTRY[name]
        MODEL_REGISTRY[name] = cls
        return cls
    return _register


def load_model(name):
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    else:
        return None
