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

# Marker for return types that include None, but where forcing the user to
# check for None can be detrimental. Sometimes called "the Any trick". See
# https://stackoverflow.com/questions/79448057/how-does-maybenone-also-known-as-the-any-trick-work-in-python-type-hints
MaybeNone: TypeAlias = Any
