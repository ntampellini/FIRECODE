from typing import Annotated, Any, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

Array3D_float = Annotated[NDArray[np.float64], "shape: (nconfs, natoms, 3)"]
Array2D_float = Annotated[NDArray[np.float64], "shape: (natoms, 3)"]
Array1D_float = Annotated[NDArray[np.float64], "shape: (n,)"]
Array2D_int = Annotated[NDArray[np.int32], "shape: (a, b)"]
Array1D_int = Annotated[NDArray[np.int_], "shape: (n,)"]
Array1D_str = Annotated[NDArray[np.str_], "shape: (nsymbols,)"]
Array1D_bool = Annotated[NDArray[np.bool_], "shape: (nbool,)"]
FloatIterable = Union[tuple[float, ...], list[float], NDArray[np.floating[Any]]]
IntIterable = Union[tuple[int, ...], list[int], NDArray[np.int_]]
MaybeNone: TypeAlias = Any
