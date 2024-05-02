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


def get_llm_class(ollm_model_id):
    """Get LLM class.

    Args:
        ollm_model_id (str): String of LLM model ID.

    Returns:
        class: LLM class.
    """
    llm_class = None
    for _, model_class in MODEL_REGISTRY.items():
        if model_class.include_name in ollm_model_id.lower():
            llm_class = model_class
    if llm_class is None:
        llm_class = MODEL_REGISTRY["default"] if "default" in MODEL_REGISTRY else None

    return llm_class
