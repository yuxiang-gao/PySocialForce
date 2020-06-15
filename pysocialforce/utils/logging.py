"""General utilities"""
import logging
from functools import wraps
from time import time

# Create a custom logger
logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)
FORMAT = "%(levelname)s:[%(filename)s:%(lineno)s %(funcName)20s() ] %(message)s"

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("file.log")
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.WARNING)

# Create formatters and add it to handlers
c_format = logging.Formatter(FORMAT)
f_format = logging.Formatter("%(asctime)s|" + FORMAT)
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.debug(f"Timeit: {f.__name__}({args}, {kw}), took: {te-ts:2.4f} sec")
        return result

    return wrap
