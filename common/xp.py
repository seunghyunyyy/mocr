# common/xp.py
# coding: utf-8
import os

USE_CUPY = os.getenv("USE_CUPY", "0") == "1"
TARGET_DTYPE = os.getenv("MOCR_DTYPE", "float16").lower()  # "float16" or "float32"

if USE_CUPY:
    import cupy as xp
    def asnumpy(x): return xp.asnumpy(x)
else:
    import numpy as xp
    def asnumpy(x): return x

DTYPE = xp.float16 if TARGET_DTYPE == "float16" else xp.float32

def backend_name() -> str:
    return "cupy" if USE_CUPY else "numpy"
