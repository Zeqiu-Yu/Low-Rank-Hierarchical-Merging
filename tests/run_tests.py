#!/usr/bin/env python
"""Run the LHM test-suite (no language model is ever loaded).

    python tests/run_tests.py            # verbose unittest run
    python tests/run_tests.py -q         # quiet

The suite covers every core module of the algorithm:

* ``test_layer_utils``  -- layer/parameter name mapping
* ``test_task_vector``  -- Phase 1 (task-vector decomposition)
* ``test_hessian``      -- Phase 2 (Hutchinson / double-backward traces)
* ``test_lsos``         -- Phase 3 (LSOS index)
* ``test_dccm``         -- Phase 4 (dual-constraint coefficient mapping)
* ``test_merging``      -- Phase 5 (layer-adaptive merging: TA/TIES/DARE/AIM)
* ``test_baselines``    -- Model Soup / TA / TIES / DARE / AIM / ACM / ORCA
* ``test_pipeline``     -- full end-to-end pipeline + IO + reporting
"""

import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
for path in (PROJECT_ROOT, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> int:
    verbosity = 1 if "-q" in sys.argv[1:] else 2
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=TESTS_DIR, pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
