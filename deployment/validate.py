"""Small real-audio ONNX parity checks; all outputs go to --save-dir."""

import argparse
from collections import Counter
import csv
import gc
import json
import pathlib
import time

import librosa
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn.utils.rnn import pad_sequence

from inference.api import load_inference_model
from inference.backend import SpectrogramContext
from inference.scoring import prepare_scoring, score_fragments, select_scored_paths
from lib.config.schema import ConfigurationScope
from lib.path_traversal import first_choices, materialize_paths
from modules.decoding import decode_alignment_flat
from modules.functional import cross_cosine_similarity


def arrays(values):
    return [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in values]


def pick_samples(data_dir, vocabulary, sample_rate):
    """Choose two whole, short, differently sized clips by CSV metadata."""
    selected = {}
    for index in sorted(data_dir.rglob("index.csv")):
        with index.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                language = row["language"]
                if language not in ("ja", "zh"):
                    continue
                duration = sum(map(float, row["durations"].split()))
                phones = row["phones"].split()
                ids = [value for p in phones if (value := vocabulary.encode(p, language)) is not None]
                if not (0.6 <= duration <= 1.8 and 3 <= len(ids) <= 16):
                    continue
                distance = abs(duration - {"ja": 1.24, "zh": 1.70}[language])
                if language in selected and distance >= selected[language][0]:
                    continue
                audio = next((index.parent / "waveforms" / (row["name"] + ext)
                              for ext in (".wav", ".flac", ".opus")
                              if (index.parent / "waveforms" / (row["name"] + ext)).is_file()), None)
                if audio is not None:
                    selected[language] = (distance, audio, ids, phones)
    if len(selected) != 2:
        raise ValueError("Need short ja and zh samples in --data-dir (0.6-1.8 seconds, 3-16 tokens).")
    samples = []
    for language in ("ja", "zh"):
        _, path, ids, phones = selected[language]
        waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
        if not 0.6 <= len(waveform) / sample_rate <= 1.85:
            raise ValueError(f"Audio duration disagrees with metadata: {path}")
        samples.append({"path": str(path.resolve()), "language": language, "phones": phones,
                        "tokens": torch.tensor(ids), "waveform": torch.from_numpy(waveform),
                        "duration": len(waveform) / sample_rate})
    return samples


class Validation:
    def __init__(self, onnx_dir, save_dir, provider, device_id):
        self.onnx_dir, self.save_dir = onnx_dir, save_dir
        self.provider, self.device_id = provider, device_id
        self.sessions = {}
        self.records = []
        self.profiles = {}
        self.started = time.monotonic()
        save_dir.mkdir(parents=True, exist_ok=True)
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"Provider unavailable: {provider}; available: {ort.get_available_providers()}")

    def session(self, name):
        if name not in self.sessions:
            path = self.onnx_dir / f"{name}.onnx"
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            if any(node.op_type in ("Loop", "Scan", "If") for node in graph.graph.node):
                raise AssertionError(f"Unexpected control flow in {path}")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.enable_profiling = True
            options.profile_file_prefix = str(self.save_dir / name)
            providers = ([('DmlExecutionProvider', {'device_id': str(self.device_id)}), 'CPUExecutionProvider']
                         if self.provider == 'DmlExecutionProvider' else ['CPUExecutionProvider'])
            session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            session.disable_fallback()
            if self.provider not in session.get_providers():
                raise AssertionError(f"Requested provider missing from {name}: {session.get_providers()}")
            self.sessions[name] = session
        return self.sessions[name]

    def run(self, name, **inputs):
        inputs = dict(zip(inputs, arrays(inputs.values())))
        return self.session(name).run(None, inputs)

    def check(self, name, actual, expected, atol=3e-4, rtol=3e-4):
        actual, expected = arrays((actual, expected))
        if actual.shape != expected.shape:
            raise AssertionError(f"{name}: {actual.shape} != {expected.shape}")
        finite = np.isfinite(expected) & np.isfinite(actual)
        error = float(np.max(np.abs(actual[finite].astype(float) - expected[finite].astype(float)), initial=0))
        exact = expected.dtype.kind in "biu"
        passed = np.array_equal(actual, expected) if exact else np.allclose(actual, expected, atol=atol, rtol=rtol)
        self.records.append({"name": name, "shape": list(actual.shape), "max_abs_error": error,
                             "passed": bool(passed), "exact": exact})
        if not passed:
            raise AssertionError(f"{name}: parity failed (max absolute error {error})")

    def compare_model(self, backend, spec, ort_spec, tokens, label):
        maskN = tokens != 0
        frames, _, phones, logits = backend.model(spec.features, tokens, spec.mask, maskN)
        reference = cross_cosine_similarity(frames, phones), logits
        actual = self.run("model", spectrogram=ort_spec[0], tokens=tokens, maskT=ort_spec[1], maskN=maskN)
        for name, value, target in zip(("similarities", "logits"), actual, reference):
            self.check(f"{label}/{name}", value, target)
        return actual, reference

    def align(self, backend, spec, ort_spec, tokens, groups, label):
        # Empty items never enter the numerical model. Restore them in the host.
        frame_lengths = torch.from_numpy(ort_spec[1].sum(-1, dtype=np.int64))
        token_lengths = (tokens != 0).long().sum(-1)
        active = (token_lengths > 0) & (frame_lengths > 0)
        if not bool(active.all()):
            raise ValueError("Use nonempty audio/token items in this bounded real-audio test.")
        actual, _ = self.compare_model(backend, spec, ort_spec, tokens, label)
        reference = backend.align(spec, tokens=tokens, groups=groups, unit="frame")
        spans = decode_alignment_flat(torch.from_numpy(actual[0]), frame_lengths, token_lengths, groups=groups)
        self.check(f"{label}/spans", spans, reference.spans)
        np.savez_compressed(self.save_dir / f"{label.replace('/', '_')}.npz",
                            tokens=tokens.numpy(), groups=groups.numpy(), spans=spans.numpy(),
                            reference_spans=reference.spans.numpy(), similarities=actual[0], logits=actual[1])
        return spans, reference.spans

    def finish(self, metadata, error=None):
        for name, session in self.sessions.items():
            path = pathlib.Path(session.end_profiling())
            events = json.loads(path.read_text(encoding="utf8"))
            counts = Counter(e.get("args", {}).get("provider") for e in events if e.get("args", {}).get("provider"))
            self.profiles[name] = {"path": str(path), "provider_events": dict(counts)}
        if self.provider == "DmlExecutionProvider" and not error:
            if not self.profiles.get("model", {}).get("provider_events", {}).get("DmlExecutionProvider", 0):
                error = "No model execution was recorded on DirectML."
        result = {**metadata, "provider": self.provider, "onnxruntime": ort.__version__,
                  "torch": torch.__version__, "elapsed_seconds": time.monotonic() - self.started,
                  "passed": error is None, "error": error, "checks": self.records, "profiles": self.profiles}
        (self.save_dir / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
        self.sessions.clear()
        gc.collect()
        if error:
            raise AssertionError(error)


def pronunciation_grid(samples):
    # Synthetic alternatives on real phonemes: shared anchors, deletion, empty
    # candidates, adjacent divergent words, fixed tokens and absent words.
    ids = (samples[0]["tokens"].tolist() + samples[1]["tokens"].tolist()) * 2
    a, b, c, d, e, f, g, h = ids[:8]
    paths = torch.tensor([
        [[h, 0, 0], [a, a, 0], [b, 0, 0], [c, c, 0], [d, e, 0], [f, f, 0], [0, g, 0], [h, 0, 0], [0, 0, 0]],
        [[h, 0, 0], [a, a, 0], [b, 0, 0], [c, c, 0], [d, e, 0], [f, 0, 0], [0, 0, 0], [h, 0, 0], [0, 0, 0]],
    ])
    words = torch.tensor([[0, 1, 1, 1, 2, 2, 2, 0, 0]]).expand(2, -1).clone()
    candidates = torch.tensor([[[True, True, True], [True, True, False], [False, False, False]]]).expand(2, -1, -1).clone()
    groups = (paths != 0).long()
    return paths, words, candidates, groups


@torch.no_grad()
def validate(args):
    torch.set_num_threads(2)
    torch.manual_seed(20260923)
    backend, vocabulary, _ = load_inference_model(args.model, scope=ConfigurationScope.FA)
    samples = pick_samples(args.data_dir, vocabulary, backend.sample_rate)
    metadata = {"checkpoint": str(args.model.resolve()), "onnx_dir": str(args.onnx_dir.resolve()),
                "max_batch_size": 2, "samples": [{k: (v.tolist() if isinstance(v, torch.Tensor) else v)
                for k, v in sample.items() if k != "waveform"} for sample in samples]}
    runner = Validation(args.onnx_dir, args.save_dir, args.provider, args.device_id)
    try:
        specs, standalone_spans = {}, {}
        for label, items in (("single_ja", [0]), ("single_zh", [1]), ("unequal_batch", [0, 1]), ("reversed_batch", [1, 0])):
            waveform = pad_sequence([samples[i]["waveform"] for i in items], batch_first=True)
            duration = torch.tensor([samples[i]["duration"] for i in items])
            tokens = pad_sequence([samples[i]["tokens"] for i in items], batch_first=True)
            groups = ((torch.arange(tokens.shape[1]) // 2 + 1).expand_as(tokens)).masked_fill(tokens == 0, 0)
            spec = backend.spectrogram(waveform, duration)
            ort_spec = runner.run("spectrogram", waveform=waveform, duration=duration)
            runner.check(f"{label}/spectrogram", ort_spec[0], spec.features, atol=5e-4)
            runner.check(f"{label}/maskT", ort_spec[1], spec.mask)
            spans, ref_spans = runner.align(backend, spec, ort_spec, tokens, groups, label)
            specs[label] = spec, ort_spec
            if len(items) == 1:
                standalone_spans[items[0]] = ref_spans[0]
            elif label == "unequal_batch":
                metadata["pytorch_single_vs_batch_max_frame_difference"] = [
                    int((standalone_spans[i] - ref_spans[b, :len(samples[i]["tokens"])]).abs().max())
                    for b, i in enumerate(items)
                ]
                batched_spans = spans
            else:
                runner.check("batch_permutation/spans", spans.flip(0), batched_spans)

        spec, ort_spec = specs["unequal_batch"]
        for unit, mixed in (("levenshtein", False), ("word", False), ("levenshtein", True)):
            paths, words, candidates, groups = pronunciation_grid(samples)
            label = f"pronunciation_{unit}" + ("_mixed" if mixed else "")
            if mixed:
                paths[1, :, 1:] = 0
                candidates[1, :, 1:] = False
            prepared = runner.run("prepare", paths=paths, words=words, candidates=candidates,
                                  grouped=np.array(unit == "word", dtype=bool))
            native = prepare_scoring(paths, words, candidates, unit=unit)
            expected = (*native, first_choices(candidates), (native[1] > 0).any(-1))
            for name, value, target in zip(("tokens", "segments", "mapping", "choices", "active"), prepared, expected):
                runner.check(f"{label}/{name}", value, target)
            masked = torch.from_numpy(prepared[0])
            active = torch.from_numpy(prepared[4])
            active_spec = SpectrogramContext(spec.features[active], spec.mask[active])
            outputs, reference = runner.compare_model(
                backend, active_spec, (ort_spec[0][prepared[4]], ort_spec[1][prepared[4]]), masked[active], label,
            )
            fragments = runner.run("score", logits=outputs[1], paths=paths[active], words=words[active],
                                   segments=prepared[1][prepared[4]], mapping=prepared[2][prepared[4]])
            ref_fragments = score_fragments(reference[1].log_softmax(-1), paths[active], words[active], native[1][active], native[2][active])
            for name, value, target in zip(("descriptors", "lengths", "costs", "tails", "capacity"), fragments, ref_fragments):
                runner.check(f"{label}/{name}", value, target)
            choices = torch.from_numpy(prepared[3].copy())
            scores = torch.zeros_like(candidates, dtype=torch.float32).masked_fill(~candidates, -torch.inf)
            choices[active], scores[active] = select_scored_paths(
                candidates[active], *map(torch.from_numpy, fragments), vocab_size=backend.model.max_vocab_size,
            )
            ref_scored = backend.score(spec, paths=paths, words=words, candidates=candidates, unit=unit)
            runner.check(f"{label}/joint_choices", choices, ref_scored.choices)
            runner.check(f"{label}/conditional_scores", scores, ref_scored.scores)
            selected = runner.run("select", paths=paths, words=words, groups=groups, choices=choices)
            target = materialize_paths(paths, words, groups, ref_scored.choices)
            for name, value, expected_value in zip(("best_tokens", "best_words", "best_groups", "maskN"), selected, (*target, target[0] != 0)):
                runner.check(f"{label}/{name}", value, expected_value)
            runner.align(backend, spec, ort_spec, torch.from_numpy(selected[0]), torch.from_numpy(selected[2]), f"{label}_aligned")
            np.savez_compressed(args.save_dir / f"{label}_choices.npz", choices=choices.numpy(), scores=scores.numpy())

        paths, words, candidates, groups = pronunciation_grid(samples)
        # Different P, C, W and B from the export examples; host skips empty MLM.
        edge_cases = [
            ("empty", torch.zeros(1, 1, 1, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1, 1, dtype=torch.bool)),
            ("empty_valid", torch.zeros(1, 1, 2, dtype=torch.long), torch.ones(1, 1, dtype=torch.long), torch.ones(1, 1, 2, dtype=torch.bool)),
            ("unambiguous", paths[:1, :5, :1], words[:1, :5], candidates[:1, :2, :1]),
            ("ties", paths[:, :4, :2], words[:, :4], candidates[:, :1, :2]),
        ]
        for label, p, w, c in edge_cases:
            for unit in ("levenshtein", "word"):
                prepared = runner.run("prepare", paths=p, words=w, candidates=c,
                                      grouped=np.array(unit == "word", dtype=bool))
                native = prepare_scoring(p, w, c, unit=unit)
                for i, expected in enumerate((*native, first_choices(c), (native[1] > 0).any(-1))):
                    runner.check(f"{label}/{unit}/prepare_{i}", prepared[i], expected)
                logits = torch.zeros(p.shape[0], p.shape[1], backend.model.max_vocab_size)
                fragments = runner.run("score", logits=logits, paths=p, words=w, segments=prepared[1], mapping=prepared[2])
                ref = score_fragments(logits.log_softmax(-1), p, w, native[1], native[2])
                choices, scores = select_scored_paths(c, *map(torch.from_numpy, fragments), vocab_size=logits.shape[-1])
                rc, rs = select_scored_paths(c, *ref, vocab_size=logits.shape[-1])
                runner.check(f"{label}/{unit}/choices", choices, rc)
                runner.check(f"{label}/{unit}/scores", scores, rs)
                selected = runner.run("select", paths=p, words=w, groups=(p != 0).long(), choices=choices)
                target = materialize_paths(p, w, (p != 0).long(), rc)
                for name, value, expected in zip(("best_tokens", "best_words", "best_groups", "maskN"), selected, (*target, target[0] != 0)):
                    runner.check(f"{label}/{unit}/{name}", value, expected)
                if label in ("empty", "empty_valid", "unambiguous"):
                    runner.check(f"{label}/{unit}/skip_mlm", prepared[4], np.zeros(p.shape[0], dtype=bool))
    except Exception as exc:
        runner.finish(metadata, error=f"{type(exc).__name__}: {exc}")
    else:
        runner.finish(metadata)
    print(f"Passed {len(runner.records)} checks on {args.provider}. Report: {args.save_dir / 'report.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--onnx-dir", type=pathlib.Path, required=True)
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data/phones"))
    parser.add_argument("--save-dir", type=pathlib.Path, required=True)
    parser.add_argument("--provider", choices=("CPUExecutionProvider", "DmlExecutionProvider"), required=True)
    parser.add_argument("--device-id", type=int, default=0)
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
