import os
_USE_CUPY = os.getenv("USE_CUPY", "0") == "1"

try:
    if _USE_CUPY:
        import cupy as xp  # GPU
    else:
        import numpy as xp  # CPU
except Exception:
    import numpy as xp      # 폴백

def asnumpy(a):
    try:
        import cupy
        if isinstance(a, cupy.ndarray):
            return cupy.asnumpy(a)
    except Exception:
        pass
    return a
