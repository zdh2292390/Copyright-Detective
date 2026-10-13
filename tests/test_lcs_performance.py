"""Compiled LCS retains original character, word and generic-token results."""
import random
import unittest
from unittest.mock import patch

from src.direct_recall import comparison


def reference_lcs(left, right):
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    current = [0] * (len(right) + 1)
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            current[j] = previous[j - 1] + 1 if left[i - 1] == right[j - 1] else max(previous[j], current[j - 1])
        previous, current = current, [0] * (len(right) + 1)
    return previous[-1]


class CollidingToken:
    def __init__(self, value):
        self.value = value
    def __hash__(self):
        return 0
    def __eq__(self, other):
        return isinstance(other, CollidingToken) and self.value == other.value


class IndexedTokenList(list):
    def __getitem__(self, index):
        return super().__getitem__(index).swapcase()


class LeftToken:
    def __eq__(self, other):
        return isinstance(other, RightToken)


class RightToken:
    def __eq__(self, other):
        return False


class LcsPerformanceTests(unittest.TestCase):
    def assert_same(self, left, right):
        self.assertEqual(comparison._lcs_length(left, right), reference_lcs(left, right))

    def test_empty_identical_and_substitution_cases(self):
        for left, right in [('', ''), ('a', ''), ('', 'b'), ('abc', 'abc'), ('ab', 'cd'), ('kitten', 'sitting'), ('abab', 'baba')]:
            with self.subTest(left=left, right=right):
                self.assert_same(left, right)
                self.assert_same(list(left), list(right))
        self.assertEqual(comparison._lcs_length('ab', 'cd'), 0)

    def test_unicode_characters_preserve_code_point_equality(self):
        for left, right in [
            ('记忆模型文档测试', '文档记忆测试'), ('😀👩\u200d🔬ée\u0301', '👩\u200d🔬ee\u0301😀'),
            ('مرحبا بالعالم', 'مرحبا عالم'), ('a\x00\ud800z', '\ud800\x00az'),
        ]:
            with self.subTest(left=repr(left), right=repr(right)):
                self.assert_same(left, right)
                self.assert_same(list(left), list(right))

    def test_word_boundaries_empty_tokens_and_repeats_are_exact(self):
        for left, right in [
            (['ab', 'c'], ['a', 'bc']), (['', 'a', ''], ['', '', 'a']),
            (['hello world', '文档', '😀', 'hello world'], ['文档', 'hello world', '😀']),
            (['a', 'ab', 'a'], ['ab', 'a', 'ab']),
        ]:
            with self.subTest(left=left, right=right):
                self.assert_same(left, right)
        self.assertEqual(comparison._lcs_length(['ab', 'c'], ['a', 'bc']), 0)

    def test_random_character_and_word_sequences_match_original(self):
        rng = random.Random(8127)
        chars = list('abc中文') + ['😀', '\u0301', '\x00']
        words = ['word', 'words', 'a b', '', '文档', '😀', 'ß', 'e\u0301']
        for index in range(600):
            tokens = chars if index % 2 else words
            left = [rng.choice(tokens) for _ in range(rng.randrange(0, 45))]
            right = [rng.choice(tokens) for _ in range(rng.randrange(0, 60))]
            with self.subTest(case=index):
                self.assert_same(left, right)
                if index % 2:
                    self.assert_same(''.join(left), ''.join(right))

    def test_generic_and_unhashable_tokens_keep_fallback_semantics(self):
        for left, right in [
            ([0, 1, 2, 3], [2, 1, 3]), ([True, None, 4.5], [1, 4.5, None]),
            ([[1], [2], [1]], [[2], [1]]), ([{'a': 1}, {'b': 2}], [{'b': 2}, {'a': 1}]),
            ((1, 2, 3), (3, 2, 1)),
            ([CollidingToken('a')], [CollidingToken('b')]),
            ([float('nan')], [float('nan')]),
        ]:
            with self.subTest(left=repr(left), right=repr(right)):
                self.assert_same(left, right)
                self.assert_same(right, left)

    def test_custom_sequence_uses_original_indexing_semantics(self):
        self.assert_same(IndexedTokenList(['a']), ['A', 'B'])
        self.assert_same(['A', 'B'], IndexedTokenList(['a']))
        self.assertEqual(comparison._lcs_length(IndexedTokenList(['a']), ['A', 'B']), 1)

    def test_fallback_transpose_preserves_equality_operand_direction(self):
        self.assert_same([LeftToken()], [RightToken(), RightToken()])
        self.assertEqual(comparison._lcs_length([LeftToken()], [RightToken(), RightToken()]), 1)
        self.assertEqual(comparison._lcs_length([RightToken(), RightToken()], [LeftToken()]), 0)

    def test_missing_or_unsupported_weighted_api_uses_original_result(self):
        for implementation in (None, lambda *args, **kwargs: (_ for _ in ()).throw(ImportError('missing')), lambda *args, **kwargs: (_ for _ in ()).throw(TypeError('weights unsupported'))):
            with self.subTest(implementation=implementation), patch.object(comparison, 'distance', implementation):
                self.assert_same(list('long string'), list('tiny'))
                self.assert_same(['a', 'whole word', 'a'], ['whole word', 'a'])

    def test_all_report_metrics_remain_identical(self):
        pairs = [
            ('This is a source passage with repeated words.', 'This is a different passage with repeated words.'),
            ('中文文档记忆测试 😀', '文档中文记忆 😀'),
            ('', ''), ('one', ''), ('a b', 'c d'),
        ]
        for reference, generated in pairs:
            with self.subTest(reference=reference, generated=generated):
                expected = None
                with patch.object(comparison, '_lcs_length', side_effect=reference_lcs):
                    expected = comparison.calculate_similarity_metrics(reference, generated)
                self.assertEqual(comparison.calculate_similarity_metrics(reference, generated), expected)


if __name__ == '__main__':
    unittest.main()