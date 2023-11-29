%pythoncode %{
import numpy as np

def __getstate__(self):
    if self.m() == 1:
        return dict(
                numpy_array=self.toNumpy1D(),
        )
    else:
        return dict(
                numpy_array=self.toNumpy2D(),
        )


def __setstate__(self, state):
    self.__init__()
    if len(state["numpy_array"].shape) == 1:
        self.fromNumpy1D(state["numpy_array"])
    else:
        self.fromNumpy2D(state["numpy_array"])


FloatCuArray.__getstate__ = __getstate__
FloatCuArray.__setstate__ = __setstate__

IntCuArray.__getstate__ = __getstate__
IntCuArray.__setstate__ = __setstate__
        %}