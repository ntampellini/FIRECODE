import logging
from typing import Generator

import pytest

from firecode.__main__ import env_variables_handling

env_variables_handling()


@pytest.fixture(autouse=True)
def restore_logging_handlers() -> Generator[None, None, None]:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    yield
    root_logger.handlers = original_handlers
