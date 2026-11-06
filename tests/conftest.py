"""Set SPARSAX_TEST_PARALLEL_BATCHES=1 to run every batch on as many threads as it has elements.

Small test systems normally stay serial, because a thread would not repay its start.
Forcing them onto threads puts the whole suite through the parallel paths, which is
what the sanitizer jobs want.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _parallel_batches():
    if os.environ.get("SPARSAX_TEST_PARALLEL_BATCHES"):
        import sparsax

        sparsax._cpp.set_batch_work_per_thread(0)
    yield
