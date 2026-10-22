"""Lightweight neural-transcription helpers for chromagram extraction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
from typing import Tuple, Union

import librosa
import numpy as np


_TRANSCRIPTION_DIR = Path(__file__).resolve().parent
_DEEP_CHROMA_DIR = _TRANSCRIPTION_DIR / "deep_chroma"
for path in (_TRANSCRIPTION_DIR, _DEEP_CHROMA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


ONSETS_AND_FRAMES_SAMPLE_RATE = 16000
ONSETS_AND_FRAMES_HOP_LENGTH = 512
MIN_MIDI = 21
N_KEYS = 88


DEEP_CHROMA_SAMPLE_RATE = 22050


def transcribe_chromagram(
    audio: np.ndarray,
    *,
    pipeline: str = "onsets_and_frames",
    checkpoint: Union[str, Path],
    device: str = "cpu",
) -> Tuple[np.ndarray, float]:
    """Transcribe audio with a neural backend and return a chromagram.

    Parameters
    ----------
    audio:
        Mono audio signal. The required sample rate depends on ``pipeline``.
    pipeline:
        Neural transcription pipeline. Options are ``"onsets_and_frames"``
        and ``"deep_chroma"``.
    checkpoint:
        Required path to a compatible checkpoint file. No default checkpoint
        location is searched.
    device:
        Torch device used for inference.

    Returns
    -------
    chromagram, feature_rate:
        Chromagram of shape ``(12, n_frames)`` and its feature rate in Hz.
    """

    pipeline = _normalize_pipeline_name(pipeline)
    checkpoint = _validate_checkpoint_path(checkpoint)
    if pipeline == "onsets_and_frames":
        return transcribe_onsets_and_frames_chromagram(
            audio,
            checkpoint=checkpoint,
            device=device,
        )
    if pipeline == "deep_chroma":
        return transcribe_deep_chroma_chromagram(
            audio,
            checkpoint=checkpoint,
            device=device,
        )

    raise ValueError('pipeline must be "onsets_and_frames" or "deep_chroma"')


def transcribe_onsets_and_frames_frame_predictions(
    audio: np.ndarray,
    *,
    checkpoint: Union[str, Path],
    device: str = "cpu",
) -> Tuple[np.ndarray, float]:
    """Return frame predictions from the onsets-and-frames model.

    The input audio must be mono and sampled at
    ``ONSETS_AND_FRAMES_SAMPLE_RATE``.

    Parameters
    ----------
    audio:
        Mono audio signal sampled at ``ONSETS_AND_FRAMES_SAMPLE_RATE``.
    checkpoint:
        Required path to a serialized onsets-and-frames transcriber checkpoint.
    device:
        Torch device used for inference.

    Returns
    -------
    frame_prediction, feature_rate:
        Frame-wise piano-roll activation matrix and its feature rate in Hz.
        The returned matrix follows the model convention ``(n_frames, 88)``.
    """

    checkpoint = _validate_checkpoint_path(checkpoint)
    audio = _validate_audio(audio)

    torch = _get_torch()
    from onsets_and_frames.mel import melspectrogram

    transcriber = _load_onsets_and_frames_transcriber(
        str(checkpoint),
        device,
    )
    melspectrogram.to(device)

    audio_tensor = torch.as_tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)
    audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)

    with torch.no_grad():
        mel = melspectrogram(
            audio_tensor.reshape(-1, audio_tensor.shape[-1])[:, :-1]
        ).transpose(-1, -2)
        _onset_pred, _offset_pred, _velocity_pred, frame_pred, *_ = transcriber(mel)

    return (
        frame_pred.detach().squeeze(0).cpu().numpy(),
        ONSETS_AND_FRAMES_SAMPLE_RATE / ONSETS_AND_FRAMES_HOP_LENGTH,
    )


def transcribe_onsets_and_frames_chromagram(
    audio: np.ndarray,
    *,
    checkpoint: Union[str, Path],
    device: str = "cpu",
) -> Tuple[np.ndarray, float]:
    """Transcribe audio using an explicitly supplied onsets-and-frames checkpoint."""

    frame_prediction, feature_rate = transcribe_onsets_and_frames_frame_predictions(
        audio,
        checkpoint=checkpoint,
        device=device,
    )
    chromagram = pianoroll_to_chromagram(frame_prediction.T)
    return chromagram, feature_rate


def transcribe_deep_chroma_chromagram(
    audio: np.ndarray,
    *,
    checkpoint: Union[str, Path],
    device: str = "cpu",
) -> Tuple[np.ndarray, float]:
    """Transcribe audio with Deep Chroma and return a chromagram.

    The input audio must be mono and sampled at ``DEEP_CHROMA_SAMPLE_RATE``.
    The model directly predicts 12 pitch-class activations.
    ``checkpoint`` must explicitly specify a compatible checkpoint file.
    """

    checkpoint = _validate_checkpoint_path(checkpoint)
    audio = _validate_audio(audio)
    torch = _get_torch()
    from libdl.data_preprocessing import compute_efficient_hcqt

    hcqt, feature_rate, _hop_length = compute_efficient_hcqt(
        audio,
        fs=DEEP_CHROMA_SAMPLE_RATE,
        fmin=librosa.note_to_hz("C1"),
        fs_hcqt_target=50,
        bins_per_octave=36,
        num_octaves=6,
        num_harmonics=5,
        num_subharmonics=1,
    )

    inputs = np.transpose(hcqt, (2, 1, 0))
    inputs = np.log1p(10 * inputs)

    context = 75
    half_context = context // 2
    inputs_padded = np.pad(
        inputs,
        ((0, 0), (half_context, half_context + 1), (0, 0)),
    )
    inputs_context = torch.from_numpy(inputs_padded).type(torch.FloatTensor).to(device)
    inputs_context.requires_grad = False

    model = _load_deep_chroma_model(str(checkpoint), device)
    max_len = int(30 * feature_rate)
    n_segments = int(np.ceil(inputs.shape[1] / max_len))

    predictions = []
    with torch.no_grad():
        for segment_index in range(n_segments):
            start_frame = segment_index * max_len
            if segment_index < n_segments - 1:
                end_frame = (segment_index + 1) * max_len + context
            else:
                end_frame = inputs_context.shape[1]

            prediction = model(inputs_context[None, :, start_frame:end_frame - 1, :])
            predictions.append(prediction[0, 0].detach().cpu().numpy())

    chromagram_frame_major = np.concatenate(predictions, axis=0)
    chromagram = chromagram_frame_major.T
    return chromagram, feature_rate


def pianoroll_to_chromagram(pianoroll: np.ndarray) -> np.ndarray:
    """Convert an 88-key piano roll to a 12-bin chromagram.

    Parameters
    ----------
    pianoroll:
        Piano-roll matrix of shape ``(88, n_frames)``. Rows are interpreted as
        consecutive MIDI pitches starting at ``MIN_MIDI``.

    Returns
    -------
    numpy.ndarray
        Chromagram of shape ``(12, n_frames)``.
    """

    pianoroll = np.asarray(pianoroll, dtype=float)
    if pianoroll.ndim != 2 or pianoroll.shape[0] != N_KEYS:
        raise ValueError(f"pianoroll must have shape ({N_KEYS}, n_frames)")

    chromagram = np.zeros((12, pianoroll.shape[1]), dtype=float)
    for piano_key in range(N_KEYS):
        pitch_class = (piano_key + MIN_MIDI) % 12
        chromagram[pitch_class] += pianoroll[piano_key]

    return chromagram


@lru_cache(maxsize=4)
def _load_onsets_and_frames_transcriber(
    checkpoint: str,
    device: str,
):
    """Load and cache an onsets-and-frames checkpoint."""

    torch = _get_torch()
    transcriber = torch.load(
        checkpoint,
        map_location=torch.device(device),
        weights_only=False,
    )
    transcriber.zero_grad()
    transcriber.eval()
    transcriber.to(device)
    return transcriber


@lru_cache(maxsize=4)
def _load_deep_chroma_model(
    checkpoint: str,
    device: str,
):
    """Load and cache a Deep Chroma checkpoint."""

    torch = _get_torch()
    from libdl.nn_models import deep_cnn_segm_sigmoid

    model = deep_cnn_segm_sigmoid(
        n_chan_input=6,
        n_chan_layers=[70, 70, 50, 10],
        n_prefilt_layers=5,
        residual=True,
        n_bins_in=6 * 12 * 3,
        n_bins_out=12,
        a_lrelu=0.3,
        p_dropout=0.2,
    )
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=torch.device(device),
        )
    )
    model.to(device)
    model.eval()
    return model


def _validate_checkpoint_path(
    checkpoint: Union[str, Path, None],
    *,
    parameter_name: str = "checkpoint",
) -> Path:
    """Require an explicit checkpoint file before audio processing or model loading."""

    if checkpoint is None or (isinstance(checkpoint, str) and not checkpoint.strip()):
        raise ValueError(
            f"{parameter_name} must be explicitly set to a compatible checkpoint file; "
            "no default checkpoint is provided"
        )
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{parameter_name} file does not exist: {path}")
    return path


def _normalize_pipeline_name(pipeline: str) -> str:
    """Return the canonical transcription pipeline name."""

    pipeline_normalized = pipeline.lower().replace("-", "_")
    if pipeline_normalized in {"onsets_and_frames", "deep_chroma"}:
        return pipeline_normalized

    raise ValueError('pipeline must be "onsets_and_frames" or "deep_chroma"')


def _validate_audio(audio: np.ndarray) -> np.ndarray:
    """Return a valid mono audio signal as ``float32``."""

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("audio must be a one-dimensional mono signal")
    if audio.size == 0:
        raise ValueError("audio is empty")
    return audio


def _get_torch():
    """Import torch lazily and raise a helpful error if it is missing."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required for neural transcription chromagram extraction"
        ) from exc

    return torch
