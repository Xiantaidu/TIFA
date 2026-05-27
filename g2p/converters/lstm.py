"""LSTM G2P converter using ONNX encoder-decoder models.
"""

import json
from pathlib import Path

import numpy as np

from g2p.registry import converter
from .paradigm import LexiconConverter


@converter(id="lstm", language=None)
class LSTMConverter(LexiconConverter):
    """Dictionary-backed G2P with LSTM ONNX model for OOV words.

    Parameters:
        *dict_path*: pronunciation dictionary (tab-separated).
        *model_path*: directory containing ``encoder.onnx``,
          ``decoder.onnx``, ``char.json``, ``phonemes.json``.
    """

    def __init__(self, *, dict_path: str = None, model_path: str) -> None:
        super().__init__(dict_path=dict_path)

        model_dir = Path(model_path)
        with open(model_dir / "char.json", "r", encoding="utf-8") as f:
            self._char_vocab: dict[str, int] = json.load(f)
        with open(model_dir / "phonemes.json", "r", encoding="utf-8") as f:
            self._phoneme_vocab: dict[str, int] = json.load(f)

        self._idx_to_phoneme: dict[int, str] = {
            v: k for k, v in self._phoneme_vocab.items()
        }
        self._unk_idx = self._phoneme_vocab["<unk>"]
        self._pad_idx = self._phoneme_vocab["<pad>"]
        self._bos_idx = self._phoneme_vocab["<bos>"]
        self._eos_idx = self._phoneme_vocab["<eos>"]
        self._char_unk_idx = self._char_vocab.get("<unk>", 0)

        self._model_path = str(model_dir)
        self._encoder_session = None
        self._decoder_session = None
        self._max_len = 48

    # ------------------------------------------------------------------
    # LexiconConverter contract
    # ------------------------------------------------------------------

    def claim(self, token: str) -> bool:
        t = token.lower()
        if t in self._dict:
            return True
        return all(c in self._char_vocab for c in t)

    def infer_oov(self, token: str) -> list[list[str]]:
        phonemes = self._predict(token)
        return [phonemes]

    # ------------------------------------------------------------------
    # ONNX inference
    # ------------------------------------------------------------------

    def _ensure_sessions(self) -> None:
        if self._encoder_session is not None:
            return
        import onnxruntime as ort

        self._encoder_session = ort.InferenceSession(
            f"{self._model_path}/encoder.onnx"
        )
        self._decoder_session = ort.InferenceSession(
            f"{self._model_path}/decoder.onnx"
        )

    def _predict(self, word: str) -> list[str]:
        self._ensure_sessions()

        word = word.lower().strip()
        indices = [
            self._char_vocab.get(c, self._char_unk_idx) for c in word
        ]
        src = np.array(
            [[self._bos_idx] + indices + [self._eos_idx]], dtype=np.int64
        )

        # Encoder
        encoder_outputs, hidden, cell = self._encoder_session.run(
            None, {"input_ids": src}
        )

        # Decoder  --  autoregressive greedy
        batch_size = 1
        decoder_input = np.full(
            (batch_size, 1), self._bos_idx, dtype=np.int64
        )
        finished = False
        predictions: list[int] = []

        for _step in range(self._max_len):
            if finished:
                break
            outputs = self._decoder_session.run(
                None,
                {
                    "decoder_input": decoder_input,
                    "hidden": hidden,
                    "cell": cell,
                    "encoder_outputs": encoder_outputs,
                },
            )
            logits, hidden, cell, _ = outputs
            pred_id = int(np.argmax(logits[0, 0, :]))
            if pred_id == self._eos_idx:
                finished = True
            else:
                predictions.append(pred_id)
            decoder_input = np.array([[pred_id]], dtype=np.int64)

        return self._decode(predictions)

    def _decode(self, pred_ids: list[int]) -> list[str]:
        result: list[str] = []
        for idx in pred_ids:
            if idx in (
                self._unk_idx,
                self._pad_idx,
                self._bos_idx,
                self._eos_idx,
            ):
                continue
            ph = self._idx_to_phoneme.get(idx)
            if ph is not None:
                result.append(ph)
        return result
