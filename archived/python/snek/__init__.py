from .snek import *  # noqa: F401,F403  (re-export the compiled extension)

from . import snek as _snek

__doc__ = _snek.__doc__
if hasattr(_snek, "__all__"):
    __all__ = _snek.__all__
