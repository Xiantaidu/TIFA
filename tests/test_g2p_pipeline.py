"""Minimal full-flow G2P pipeline tests (programmatic + config-based)."""

import unittest
from pathlib import Path

from g2p import G2PPipeline
from g2p.api import build_pipeline_from_config
from g2p.tokenizers.cjk import CJKTokenizer
from g2p.preprocessors.simple import StripWhitespacePreprocessor
from g2p.converters.mandarin import MandarinConverter
from g2p.converters.cantonese import CantoneseConverter
from g2p.converters.simple import PassthroughConverter
from lib.config.schema import G2PPipelineConfig


class TestG2PPipeline(unittest.TestCase):
    """Full G2P pipeline: preprocess → tokenize → convert."""

    @classmethod
    def setUpClass(cls):
        cls.cmn = G2PPipeline(
            preprocessors=[StripWhitespacePreprocessor()],
            tokenizers=[CJKTokenizer()],
            converters=[MandarinConverter(), PassthroughConverter()],
        )
        cls.yue = G2PPipeline(
            preprocessors=[StripWhitespacePreprocessor()],
            tokenizers=[CJKTokenizer()],
            converters=[CantoneseConverter(), PassthroughConverter()],
        )

    # ---- Mandarin ----

    def test_mandarin_simple(self):
        self.assertEqual(self.cmn.convert("你好"), [["ni"], ["hao"]])

    def test_mandarin_polyphonic(self):
        self.assertEqual(
            self.cmn.convert("银行行长"),
            [["yin"], ["hang"], ["hang"], ["zhang"]],
        )

    def test_mandarin_traditional(self):
        self.assertEqual(self.cmn.convert("魚"), [["yu"]])

    def test_mandarin_mixed(self):
        self.assertEqual(
            self.cmn.convert("Hello你好123"),
            [["Hello"], ["ni"], ["hao"], ["123"]],
        )

    def test_mandarin_whitespace_stripped(self):
        self.assertEqual(self.cmn.convert("  你好  "), [["ni"], ["hao"]])

    # ---- Cantonese ----

    def test_cantonese_simple(self):
        self.assertEqual(self.yue.convert("你好"), [["nei"], ["hou"]])

    def test_cantonese_phrase(self):
        self.assertEqual(
            self.yue.convert("大排檔"),
            [["daai"], ["paai"], ["dong"]],
        )


class TestG2PBuildFromConfig(unittest.TestCase):
    """Config-based pipeline construction (G2P.md §Programmatic usage)."""

    def test_build_mandarin_from_config(self):
        config = G2PPipelineConfig.model_validate({
            "preprocessors": [{"id": "strip_whitespace"}],
            "tokenizers": [{"id": "cjk"}],
            "converters": [{"id": "mandarin"}, {"id": "passthrough"}],
        })
        pipeline = build_pipeline_from_config(config)
        self.assertEqual(pipeline.convert("你好"), [["ni"], ["hao"]])

    def test_build_cantonese_from_config(self):
        config = G2PPipelineConfig.model_validate({
            "tokenizers": [{"id": "cjk"}],
            "converters": [{"id": "cantonese"}, {"id": "passthrough"}],
        })
        pipeline = build_pipeline_from_config(config)
        self.assertEqual(pipeline.convert("你好"), [["nei"], ["hou"]])


class TestG2PLanguageFiltering(unittest.TestCase):
    """Language filtering with multiple converters (G2P.md §Language filtering)."""

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

    def test_filter_mandarin(self):
        self.assertEqual(
            self.pipeline.convert("你", languages=["cmn"]), [["ni"]],
        )

    def test_filter_cantonese(self):
        self.assertEqual(
            self.pipeline.convert("你", languages=["yue"]), [["nei"]],
        )

    def test_filter_no_match_falls_to_passthrough(self):
        # PassthroughConverter (language=None) always runs
        self.assertEqual(
            self.pipeline.convert("你", languages=["eng"]), [["你"]],
        )


if __name__ == "__main__":
    unittest.main()
