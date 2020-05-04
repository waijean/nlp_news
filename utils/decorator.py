import functools
import inspect
import logging
import time
from typing import List, Union


def logging_time(level: int = logging.DEBUG, args_to_log=[], kwargs_to_log=[]):
    def decorator(method):
        logger = logging.getLogger(inspect.getmodule(method).__name__)

        @functools.wraps(method)
        def _wrapper(*args, **kwargs):
            logged_args = f"{get_arg_to_str(args, args_to_log)}{get_kwargs_to_str(kwargs, kwargs_to_log)} "
            ts = time.time()
            logger.log(
                level=level, msg=f"{method.__name__} {logged_args}: logging_time"
            )
            result = method(*args, **kwargs)
            logger.log(
                level=level,
                msg=f"{method.__name__} : time taken: {round((time.time() - ts) * 1000, 1)} ms",
            )
            return result

        return _wrapper

    return decorator


def get_arg_to_str(args, args_to_log) -> str:
    return f"({[args[index] for index in args_to_log]})" if len(args_to_log) > 0 else ""


def get_kwargs_to_str(kwargs, kwargs_to_log) -> Union[List[str], str]:
    if len(kwargs_to_log) > 0:
        return [f"{key}={kwargs[key]}" for key in kwargs_to_log]
    else:
        return ""
