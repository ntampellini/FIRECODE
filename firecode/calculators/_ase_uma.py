import os

from firecode.settings import UMA_MODEL_PATH


def get_uma_calc(method="omol", logfunction=None):
    """Load UMA model from disk and return the ASE calculator object"""
    try:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        from torch import cuda

    except ImportError as err:
        print(err)
        raise ImportError(
            "To run the UMA models, please install fairchem:\n"
            "    >>> uv pip install fairchem-core\n"
            'or alternatively, install the "uma" version of firecode:\n'
            "    >>> uv pip install firecode[uma]\n"
        )

    gpu_bool = cuda.is_available()

    if gpu_bool:
        if logfunction is not None:
            logfunction(f"--> {cuda.device_count()} CUDA devices detected: loading model on GPU")

    elif logfunction is not None:
        logfunction("--> No CUDA devices detected: loading model on CPU")

    if logfunction is not None:
        logfunction(f"--> Loading UMA/{method.upper()} model from file")

    if UMA_MODEL_PATH[0] == ".":
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), os.path.basename(UMA_MODEL_PATH)
        )
    else:
        path = UMA_MODEL_PATH

    try:
        predictor = load_predict_unit(path, device="cuda" if gpu_bool else "cpu")

    except FileNotFoundError:
        raise FileNotFoundError(f'Invalid path for UMA model: UMA_MODEL_PATH="{path}".')

    ase_calc = FAIRChemCalculator(predictor, task_name=method.lower())
    return ase_calc
