import gc
from collections import UserDict
from contextlib import ContextDecorator
from functools import wraps

import torch

model_cache = UserDict(
    dict(
        preloaded_model_id=None,
        preloaded_model=None,
        preloaded_tokenizer=None,
        preloaded_streamer=None,
        preloaded_device=None,
    ))


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def clear_cache():
    gc.collect()
    torch_gc()


def clear_cache_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        clear_cache()
        res = func(*args, **kwargs)
        clear_cache()
        return res
    return wrapper


class ClearCacheContext(ContextDecorator):
    def __enter__(self):
        clear_cache()
        return self

    def __exit__(self, *exc):
        clear_cache()
        return False
