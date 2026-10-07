"""Administrator controls for bounded document chunk scheduling."""

import unittest
from unittest.mock import Mock, patch

from src.document_jobs import DocumentAnalysisJobs


class DocumentConcurrencyLimitTests(unittest.TestCase):
    def test_default_threshold_and_global_pool_bound_the_document_window(self):
        with patch.dict("os.environ", {
            "COPYRIGHT_DETECTIVE_DOCUMENT_CHUNK_CONCURRENCY": "3",
            "COPYRIGHT_DETECTIVE_DOCUMENT_CHUNK_WORKERS": "2",
        }):
            manager = DocumentAnalysisJobs(store=Mock())
        try:
            self.assertEqual(manager.concurrency_for(7), 1)
            self.assertEqual(manager.concurrency_for(8), 2)
            self.assertEqual(manager.concurrency_for(1974), 2)
        finally:
            manager.close()

    def test_administrator_can_disable_parallelism_for_a_large_document(self):
        with patch.dict("os.environ", {
            "COPYRIGHT_DETECTIVE_DOCUMENT_CHUNK_CONCURRENCY": "1",
            "COPYRIGHT_DETECTIVE_DOCUMENT_CHUNK_WORKERS": "8",
        }):
            manager = DocumentAnalysisJobs(store=Mock())
        try:
            self.assertEqual(manager.concurrency_for(1974), 1)
        finally:
            manager.close()

    def test_invalid_overrides_fall_back_to_safe_defaults(self):
        for invalid in ("", "invalid", "0", "-1", "999999"):
            with self.subTest(invalid=invalid), patch.dict("os.environ", {
                "COPYRIGHT_DETECTIVE_DOCUMENT_CHUNK_CONCURRENCY": invalid,
                "COPYRIGHT_DETECTIVE_DOCUMENT_CHUNK_WORKERS": invalid,
            }):
                manager = DocumentAnalysisJobs(store=Mock())
                try:
                    self.assertEqual(manager.concurrency_for(1974), 3)
                    self.assertEqual(manager.max_chunk_workers, 8)
                finally:
                    manager.close()


if __name__ == "__main__":
    unittest.main()
