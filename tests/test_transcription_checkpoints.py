"""Checkpoint configuration must be explicit for every neural entry point."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import tovis
from tovis.transcription import transcribe as transcription


class CheckpointConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.audio = np.ones(32)
        self.direct_functions = (
            transcription.transcribe_chromagram,
            transcription.transcribe_onsets_and_frames_frame_predictions,
            transcription.transcribe_onsets_and_frames_chromagram,
            transcription.transcribe_deep_chroma_chromagram,
        )

    def test_direct_functions_require_checkpoint_keyword(self):
        for function in self.direct_functions:
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(TypeError, "checkpoint"):
                    function(self.audio)

    def test_direct_functions_reject_empty_paths_before_importing_torch(self):
        with patch.object(transcription, "_get_torch") as get_torch:
            for function in self.direct_functions:
                for checkpoint in (None, "", "   "):
                    with self.subTest(function=function.__name__, checkpoint=checkpoint):
                        with self.assertRaisesRegex(ValueError, "checkpoint must be explicitly set"):
                            function(self.audio, checkpoint=checkpoint)
            get_torch.assert_not_called()

    def test_direct_functions_reject_missing_files_and_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(transcription, "_get_torch") as get_torch:
                for function in self.direct_functions:
                    for checkpoint in (Path(directory), Path(directory) / "missing.pt"):
                        with self.subTest(function=function.__name__, checkpoint=checkpoint):
                            with self.assertRaisesRegex(FileNotFoundError, "checkpoint file"):
                                function(self.audio, checkpoint=checkpoint)
                get_torch.assert_not_called()

    def test_feature_functions_require_paths_before_loading_or_resampling_audio(self):
        with patch("tovis.feature.librosa.load") as load_audio:
            with patch("tovis.feature.librosa.resample") as resample:
                for backend in ("ONSETS_AND_FRAMES", "DEEP_CHROMA"):
                    for function, source in (
                        (tovis.compute_chromagram, self.audio),
                        (tovis.compute_chromagram_from_file, "unused.wav"),
                    ):
                        with self.subTest(backend=backend, function=function.__name__):
                            with self.assertRaisesRegex(ValueError, "transcription_checkpoint"):
                                function(source, backend=backend)
                            for checkpoint in (None, "", "   "):
                                with self.assertRaisesRegex(ValueError, "transcription_checkpoint"):
                                    function(source, backend=backend, transcription_checkpoint=checkpoint)
                load_audio.assert_not_called()
                resample.assert_not_called()

    def test_feature_functions_reject_missing_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pt"
            with patch("tovis.feature.librosa.load") as load_audio:
                for backend in ("ONSETS_AND_FRAMES", "DEEP_CHROMA"):
                    for function, source in (
                        (tovis.compute_chromagram, self.audio),
                        (tovis.compute_chromagram_from_file, "unused.wav"),
                    ):
                        with self.subTest(backend=backend, function=function.__name__):
                            with self.assertRaisesRegex(FileNotFoundError, "transcription_checkpoint"):
                                function(source, backend=backend, transcription_checkpoint=missing)
                load_audio.assert_not_called()

    def test_feature_functions_forward_the_configured_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "user-model.pt"
            checkpoint.touch()
            for backend, pipeline, sample_rate in (
                ("ONSETS_AND_FRAMES", "onsets_and_frames", 16000),
                ("DEEP_CHROMA", "deep_chroma", 22050),
            ):
                for function, source in (
                    (tovis.compute_chromagram, self.audio),
                    (tovis.compute_chromagram_from_file, "unused.wav"),
                ):
                    with self.subTest(backend=backend, function=function.__name__):
                        with patch.object(
                            transcription, "transcribe_chromagram",
                            return_value=(np.ones((12, 2)), 10.0),
                        ) as transcribe:
                            with patch("tovis.feature.librosa.load", return_value=(self.audio, sample_rate)):
                                function(source, sample_rate=sample_rate, backend=backend,
                                         transcription_checkpoint=checkpoint)
                        self.assertEqual(transcribe.call_args.kwargs["checkpoint"], checkpoint.resolve())
                        self.assertEqual(transcribe.call_args.kwargs["pipeline"], pipeline)

    def test_dispatcher_forwards_explicit_paths_to_both_pipelines(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "user-model.pt"
            checkpoint.touch()
            for pipeline, name in (
                ("onsets_and_frames", "transcribe_onsets_and_frames_chromagram"),
                ("deep_chroma", "transcribe_deep_chroma_chromagram"),
            ):
                with self.subTest(pipeline=pipeline):
                    with patch.object(transcription, name) as backend:
                        transcription.transcribe_chromagram(
                            self.audio, pipeline=pipeline, checkpoint=checkpoint, device="cpu"
                        )
                        backend.assert_called_once_with(
                            self.audio, checkpoint=checkpoint.resolve(), device="cpu"
                        )


if __name__ == "__main__":
    unittest.main()
