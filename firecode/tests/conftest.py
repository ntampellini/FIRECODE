import logging
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def restore_logging_handlers() -> Generator[None, None, None]:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    yield
    root_logger.handlers = original_handlers
