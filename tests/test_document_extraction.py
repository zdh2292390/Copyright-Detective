"""Extraction regressions for blank PDF pages and text encoding boundaries."""

import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import src.direct_recall.pdf_utils as documents


class DocumentExtractionTests(unittest.TestCase):
    def test_blank_pdf_pages_do_not_abort_extraction_or_merge_words(self):
        pages = [SimpleNamespace(extract_text=lambda: "first page"),
                 SimpleNamespace(extract_text=lambda: None),
                 SimpleNamespace(extract_text=lambda: "last page")]
        with patch.object(documents.PyPDF2, "PdfReader", return_value=SimpleNamespace(pages=pages)):
            text = documents.extract_text_from_pdf(io.BytesIO(b"pdf"))
        self.assertEqual(text.split(), ["first", "page", "last", "page"])
        self.assertNotIn("pagelast", text)

    def test_scanned_pdf_with_no_text_returns_empty_for_ui_validation(self):
        pages = [SimpleNamespace(extract_text=lambda: None)]
        with patch.object(documents.PyPDF2, "PdfReader", return_value=SimpleNamespace(pages=pages)):
            self.assertEqual(documents.extract_text_from_pdf(io.BytesIO(b"pdf")), "")

    def test_utf8_bom_is_removed_before_chunking(self):
        self.assertEqual(documents.extract_text_from_txt(io.BytesIO(b"\xef\xbb\xbffirst word")), "first word")

    def test_utf16_and_multilingual_utf8_remain_readable(self):
        for encoding in ("utf-8", "utf-16"):
            with self.subTest(encoding=encoding):
                text = "中文文本 with words"
                self.assertEqual(documents.extract_text_from_txt(io.BytesIO(text.encode(encoding))), text)


if __name__ == "__main__":
    unittest.main()
