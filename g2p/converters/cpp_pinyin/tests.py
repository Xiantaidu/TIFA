"""Tests for the cpp-pinyin G2P converter integration.

Verifies that preprocessing and tokenization use the project's own components,
and the cpp-pinyin engine operates solely as the converter stage.
"""

import unittest

from g2p import G2PPipeline
from g2p.preprocessors.simple import LowercasePreprocessor
from g2p.tokenizers.cjk import CJKTokenizer
from g2p.converters.mandarin import MandarinConverter
from g2p.converters.cantonese import CantoneseConverter
from g2p.converters.simple import PassthroughConverter


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
        self.assertEqual(result, [["ni"], ["hao"]])

    def test_mandarin_phrase_disambiguation(self):
        result = self.mandarin.convert("一了百了")
        self.assertEqual(result, [["yi"], ["liao"], ["bai"], ["liao"]])

    def test_consecutive_phrases(self):
        result = self.mandarin.convert("一了百了一个半")
        self.assertEqual(result, [
            ["yi"], ["liao"], ["bai"], ["liao"],
            ["yi"], ["ge"], ["ban"],
        ])

    def test_polyphonic_classic(self):
        result = self.mandarin.convert("银行行长")
        self.assertEqual(result, [
            ["yin"], ["hang"], ["hang"], ["zhang"],
        ])

    def test_traditional_to_simplified(self):
        result = self.mandarin.convert("魚")
        self.assertEqual(result, [["yu"]])

    def test_cantonese_simple(self):
        result = self.cantonese.convert("你好")
        self.assertEqual(result, [["nei"], ["hou"]])

    def test_cantonese_phrase(self):
        result = self.cantonese.convert("大排檔")
        self.assertEqual(result, [["daai"], ["paai"], ["dong"]])

    def test_mixed_language(self):
        result = self.mixed.convert("Hello你好World")
        # Non-CJK handled by PassthroughConverter, CJK by MandarinConverter
        self.assertEqual(result, [
            ["hello"], ["ni"], ["hao"], ["world"],
        ])

    def test_non_cjk_passthrough(self):
        result = self.mixed.convert("Hello123")
        self.assertEqual(result, [["hello123"]])


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
        self.assertEqual(result, [["ni"]])

    def test_cantonese_filter(self):
        result = self.pipeline.convert("你", languages=["yue"])
        self.assertEqual(result, [["nei"]])

    def test_english_no_filter(self):
        result = self.pipeline.convert("Hello")
        self.assertEqual(result, [["Hello"]])


if __name__ == "__main__":
    unittest.main()
