"""Document preview/execution chunk-count and overlap regressions."""

import unittest

from src.direct_recall.pdf_utils import split_text_into_chunks
from src.pages.document_memorization_detection import _document_cache_id


class DocumentChunkingTests(unittest.TestCase):
    def test_200_words_and_50_overlap_produce_the_expected_1974_pairs(self):
        pairs = split_text_into_chunks("word " * 296101, chunk_size=200, overlap=50)
        self.assertEqual(len(pairs), 1974)
        self.assertEqual(len(pairs[0][0].split()), 200)
        self.assertEqual(len(pairs[-1][1].split()), 1)

    def test_overlapping_windows_preserve_document_order(self):
        text = " ".join(str(index) for index in range(400))
        pairs = split_text_into_chunks(text, chunk_size=200, overlap=50)
        self.assertEqual(len(pairs), 2)
        prefix, target = pairs[0]
        self.assertEqual(prefix.split()[-50:], target.split()[:50])
        self.assertEqual(target.split()[0], "150")
        self.assertEqual(pairs[-1][1].split()[-1], "399")

    def test_overlap_equal_to_chunk_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than"):
            split_text_into_chunks("word " * 200, chunk_size=50, overlap=50)

    def test_same_name_and_size_documents_do_not_reuse_text_preview(self):
        class Document:
            name = "same.txt"
            def __init__(self, data):
                self.data = data
            def getvalue(self):
                return self.data
        self.assertNotEqual(
            _document_cache_id(Document(b"first")),
            _document_cache_id(Document(b"other")),
        )


if __name__ == "__main__":
    unittest.main()
