MODEL_REGISTRY = {}
CPP_MODEL_REGISTRY = {}


def register_model(name):

    def _register(cls):
        if name in MODEL_REGISTRY:
            return MODEL_REGISTRY[name]
        MODEL_REGISTRY[name] = cls
        return cls
    return _register


def register_cpp_model(name):

    def _register(cls):
        if name in CPP_MODEL_REGISTRY:
            return CPP_MODEL_REGISTRY[name]
        CPP_MODEL_REGISTRY[name] = cls
        return cls
    return _register


def load_model(name):
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    else:
        return None


def load_cpp_model(name):
    if name in CPP_MODEL_REGISTRY:
        return CPP_MODEL_REGISTRY[name]
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


def get_cpp_llm_class(cpp_ollm_model_id):
    """Get C++ LLM class.

    Args:
        cpp_ollm_model_id (str): String of C++ LLM model ID.

    Returns:
        class: C++ LLM class.
    """
    llm_class = None
    for _, model_class in CPP_MODEL_REGISTRY.items():
        if model_class.include_name in cpp_ollm_model_id.lower():
            llm_class = model_class
    if llm_class is None:
        llm_class = CPP_MODEL_REGISTRY["default"] if "default" in CPP_MODEL_REGISTRY else None

    return llm_class
