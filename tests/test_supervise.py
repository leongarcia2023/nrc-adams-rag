import unittest
from adams_rag.supervise import should_restart


class SupervisorTests(unittest.TestCase):
    def test_restarts_only_unexpected_terminations_with_a_bounded_budget(self):
        self.assertTrue(should_restart(-9, "running", False, 0))
        self.assertTrue(should_restart(-15, "connection_wait", False, 2))
        self.assertFalse(should_restart(-9, "running", False, 3))
        self.assertFalse(should_restart(1, "running", False, 0))
        self.assertFalse(should_restart(0, "running", False, 0))

    def test_stop_completion_storage_and_validation_blocks_are_respected(self):
        self.assertFalse(should_restart(-9, "running", True, 0))
        for state in ("stopped", "finished", "needs_storage", "needs_review"):
            self.assertFalse(should_restart(-9, state, False, 0))
