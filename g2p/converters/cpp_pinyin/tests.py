"""Tests for the cpp-pinyin G2P converter integration.

Verifies that preprocessing and tokenization use the project's own components,
and the cpp-pinyin engine operates solely as the converter stage.
"""

import unittest

from g2p import G2PPipeline
from g2p.converters.base import PronunciationGroup
from g2p.converters.chinese import CantoneseConverter, MandarinConverter
from g2p.converters.simple import PassthroughConverter
from g2p.preprocessors.simple import LowercasePreprocessor
from g2p.tokenizers.cjk import CJKTokenizer


class TestCppPinyinIntegration(unittest.TestCase):
    """Full G2P pipeline integration: project preprocess/tokenize + cpp-pinyin convert."""

    @classmethod
    def setUpClass(cls):
        cls.mandarin = G2PPipeline(
            tokenizers=[CJKTokenizer()],
            converters=[MandarinConverter(), PassthroughConverter()],
        )
        cls.cantonese = G2PPipeline(
            tokenizers=[CJKTokenizer()],
            converters=[CantoneseConverter(), PassthroughConverter()],
        )
        cls.mixed = G2PPipeline(
            preprocessors=[LowercasePreprocessor()],
            tokenizers=[CJKTokenizer()],
            converters=[MandarinConverter(), PassthroughConverter()],
        )

    def test_mandarin_simple(self):
        result = self.mandarin.convert("你好")
        self.assertEqual(result, [
            PronunciationGroup(paths=[["ni"]]),
            PronunciationGroup(paths=[["hao"]]),
        ])

    def test_mandarin_phrase_disambiguation(self):
        result = self.mandarin.convert("一了百了")
        self.assertEqual(result, [
            PronunciationGroup(paths=[["yi"]]),
            PronunciationGroup(paths=[["liao"], ["le"]]),
            PronunciationGroup(paths=[["bai"]]),
            PronunciationGroup(paths=[["liao"], ["le"]]),
        ])

    def test_consecutive_phrases(self):
        result = self.mandarin.convert("一了百了一个半")
        self.assertEqual(result, [
            PronunciationGroup(paths=[["yi"]]),
            PronunciationGroup(paths=[["liao"], ["le"]]),
            PronunciationGroup(paths=[["bai"]]),
            PronunciationGroup(paths=[["liao"], ["le"]]),
            PronunciationGroup(paths=[["yi"]]),
            PronunciationGroup(paths=[["ge"]]),
            PronunciationGroup(paths=[["ban"]]),
        ])

    def test_polyphonic_classic(self):
        result = self.mandarin.convert("银行行长")
        self.assertEqual(result, [
            PronunciationGroup(paths=[["yin"]]),
            PronunciationGroup(paths=[["hang"], ["xing"], ["heng"]]),
            PronunciationGroup(paths=[["hang"], ["xing"], ["heng"]]),
            PronunciationGroup(paths=[["zhang"], ["chang"]]),
        ])

    def test_traditional_to_simplified(self):
        result = self.mandarin.convert("魚")
        self.assertEqual(result, [
            PronunciationGroup(paths=[["yu"]]),
        ])

    def test_cantonese_simple(self):
        result = self.cantonese.convert("你好")
        self.assertEqual(result, [
            PronunciationGroup(paths=[["nei"]]),
            PronunciationGroup(paths=[["hou"]]),
        ])

    def test_cantonese_phrase(self):
        result = self.cantonese.convert("大排檔")
        self.assertEqual(result, [
            PronunciationGroup(paths=[["daai"]]),
            PronunciationGroup(paths=[["paai"]]),
            PronunciationGroup(paths=[["dong"]]),
        ])

    def test_mixed_language(self):
        result = self.mixed.convert("Hello你好World")
        # Non-CJK handled by PassthroughConverter, CJK by MandarinConverter
        self.assertEqual(result, [
            PronunciationGroup(paths=[["hello"]]),
            PronunciationGroup(paths=[["ni"]]),
            PronunciationGroup(paths=[["hao"]]),
            PronunciationGroup(paths=[["world"]]),
        ])

    def test_non_cjk_passthrough(self):
        result = self.mixed.convert("Hello123")
        self.assertEqual(result, [
            PronunciationGroup(paths=[["hello123"]]),
        ])


class TestLanguageFiltering(unittest.TestCase):
    """Language filtering between Mandarin and Cantonese converters."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = G2PPipeline(
            tokenizers=[CJKTokenizer()],
            converters=[
                MandarinConverter(),
                CantoneseConverter(),
                PassthroughConverter(),
            ],
        )

    def test_mandarin_filter(self):
        result = self.pipeline.convert("你", languages=["cmn"])
        self.assertEqual(result, [PronunciationGroup(paths=[["ni"]])])

    def test_cantonese_filter(self):
        result = self.pipeline.convert("你", languages=["yue"])
        self.assertEqual(result, [PronunciationGroup(paths=[["nei"]])])

    def test_english_no_filter(self):
        result = self.pipeline.convert("Hello")
        self.assertEqual(result, [PronunciationGroup(paths=[["Hello"]])])


if __name__ == "__main__":
    unittest.main()
