"""Feature extraction utilities for TOVIS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd

from .utils import normalize_feature_sequence, read_csv, smooth_resample_feature_sequence


ChromaBackend = str
NormType = str


def compute_chromagram(
    audio: np.ndarray,
    sample_rate: int = 22050,
    *,
    backend: ChromaBackend = "CQT",
    window_length: int = 4096,
    hop_length: int = 2048,
    estimate_tuning: bool = False,
    gamma: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    transcription_checkpoint: Optional[Union[str, Path]] = None,
    transcription_device: str = "cpu",
    smoothing_length_seconds: Optional[float] = None,
    output_feature_rate: Optional[float] = None,
    smoothing_window_type: str = "boxcar",
    norm: Optional[NormType] = "2",
    norm_threshold: float = 1e-3,
) -> Tuple[np.ndarray, float]:
    """Compute a 12-bin chromagram from a mono audio signal.

    Parameters
    ----------
    audio:
        Mono audio signal.
    sample_rate:
        Audio sample rate in Hz.
    backend:
        Chroma backend. Options are ``"STFT"``, ``"CQT"``, ``"IIR"``,
        ``"VQT"``, ``"ONSETS_AND_FRAMES"``, and ``"DEEP_CHROMA"``.
    window_length:
        Analysis window length for STFT and IIR features.
    hop_length:
        Hop length in audio samples.
    estimate_tuning:
        If true, estimate global tuning and pass it to the chroma backend.
    gamma:
        Optional logarithmic compression, applied as ``log(1 + gamma * X)``.
    start_time:
        Optional start time in seconds for all backends.
    end_time:
        Optional end time in seconds for all backends.
    transcription_checkpoint:
        Required checkpoint file path for ``backend="ONSETS_AND_FRAMES"`` or
        ``backend="DEEP_CHROMA"``. No default location is searched. Unused
        for other backends.
    transcription_device:
        Torch device for transcription.
    smoothing_length_seconds:
        Optional temporal smoothing-window duration in seconds. The nearest
        integer number of source feature frames is used; see
        :func:`~tovis.utils.smooth_resample_feature_sequence`.
    output_feature_rate:
        Optional output frame rate in Hz. The chromagram is linearly
        interpolated onto a regular grid at exactly this rate; see
        :func:`~tovis.utils.smooth_resample_feature_sequence`.
    smoothing_window_type:
        Window type passed to
        :func:`~tovis.utils.smooth_resample_feature_sequence`.
    norm:
        Optional frame-wise output normalization. Supports ``"1"``, ``"2"``,
        ``"max"``, and ``"z"``; see
        :func:`~tovis.utils.normalize_feature_sequence`.
    norm_threshold:
        Threshold passed to :func:`~tovis.utils.normalize_feature_sequence`.

    Returns
    -------
    chromagram, feature_rate:
        Chromagram of shape ``(12, n_frames)`` and its frame rate in Hz.
    """

    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 1:
        raise ValueError("audio must be a one-dimensional mono signal")

    backend = _normalize_backend_name(backend)
    audio = _slice_audio(audio, sample_rate=sample_rate, start_time=start_time, end_time=end_time)
    tuning = 0.0
    if backend not in {"ONSETS_AND_FRAMES", "DEEP_CHROMA"}:
        tuning = librosa.estimate_tuning(y=audio, sr=sample_rate) if estimate_tuning else 0.0

    if backend == "STFT":
        spectrogram = librosa.stft(
            audio,
            n_fft=window_length,
            hop_length=hop_length,
            center=True,
            pad_mode="constant",
        )
        power_spectrogram = np.abs(spectrogram) ** 2
        chromagram = librosa.feature.chroma_stft(
            S=power_spectrogram,
            sr=sample_rate,
            tuning=tuning,
            norm=None,
            hop_length=hop_length,
            n_fft=window_length,
        )
    elif backend == "CQT":
        chromagram = librosa.feature.chroma_cqt(
            y=audio,
            sr=sample_rate,
            hop_length=hop_length,
            tuning=tuning,
            norm=None,
        )
    elif backend == "IIR":
        time_frequency = librosa.iirt(
            y=audio,
            sr=sample_rate,
            win_length=window_length,
            hop_length=hop_length,
            center=True,
            tuning=0.0,
        )
        chromagram = librosa.feature.chroma_cqt(
            C=time_frequency,
            bins_per_octave=12,
            n_octaves=7,
            fmin=librosa.midi_to_hz(24),
            tuning=tuning,
            norm=None,
        )
    elif backend == "VQT":
        chromagram = librosa.feature.chroma_vqt(
            y=audio,
            sr=sample_rate,
            hop_length=hop_length,
            intervals="equal",
            norm=None,
        )
    elif backend in {"ONSETS_AND_FRAMES", "DEEP_CHROMA"}:
        from .transcription.transcribe import (
            DEEP_CHROMA_SAMPLE_RATE,
            ONSETS_AND_FRAMES_SAMPLE_RATE,
            _validate_checkpoint_path,
            transcribe_chromagram,
        )

        checkpoint = _validate_checkpoint_path(
            transcription_checkpoint, parameter_name="transcription_checkpoint"
        )
        if backend == "ONSETS_AND_FRAMES":
            target_sample_rate = ONSETS_AND_FRAMES_SAMPLE_RATE
            pipeline = "onsets_and_frames"
        else:
            target_sample_rate = DEEP_CHROMA_SAMPLE_RATE
            pipeline = "deep_chroma"

        if sample_rate != target_sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=target_sample_rate,
            )
            sample_rate = target_sample_rate

        chromagram, feature_rate = transcribe_chromagram(
            audio,
            pipeline=pipeline,
            checkpoint=checkpoint,
            device=transcription_device,
        )
    else:
        raise ValueError(f"Unsupported chroma backend: {backend}")

    if backend not in {"ONSETS_AND_FRAMES", "DEEP_CHROMA"}:
        feature_rate = sample_rate / hop_length

    if gamma is not None:
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        chromagram = np.log1p(gamma * chromagram)

    if smoothing_length_seconds is not None or output_feature_rate is not None:
        chromagram, feature_rate = smooth_resample_feature_sequence(
            chromagram,
            feature_rate=feature_rate,
            smoothing_length_seconds=smoothing_length_seconds,
            output_feature_rate=output_feature_rate,
            smoothing_window_type=smoothing_window_type,
        )

    if norm is not None:
        chromagram = normalize_feature_sequence(
            chromagram,
            norm=norm,
            threshold=norm_threshold,
        )

    return chromagram, feature_rate


def compute_chromagram_from_file(
    filename: Union[str, Path],
    sample_rate: int = 22050,
    *,
    backend: ChromaBackend = "CQT",
    window_length: int = 4096,
    hop_length: int = 2048,
    estimate_tuning: bool = False,
    gamma: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    transcription_checkpoint: Optional[Union[str, Path]] = None,
    transcription_device: str = "cpu",
    smoothing_length_seconds: Optional[float] = None,
    output_feature_rate: Optional[float] = None,
    smoothing_window_type: str = "boxcar",
    norm: Optional[NormType] = "2",
    norm_threshold: float = 1e-3,
) -> Tuple[np.ndarray, float]:
    """Load an audio file and compute a chromagram.

    Audio is loaded with ``librosa.load`` and passed to ``compute_chromagram``.
    If ``start_time`` and/or ``end_time`` are given, only that time span is
    loaded from disk.

    Parameters
    ----------
    filename:
        Path to the audio file.
    sample_rate:
        Input sample rate for librosa chroma backends.
    backend:
        Chroma backend passed to ``compute_chromagram``.
    window_length:
        Analysis window length for STFT and IIR features.
    hop_length:
        Hop length in audio samples.
    estimate_tuning:
        If true, estimate global tuning for librosa chroma backends.
    gamma:
        Optional logarithmic compression factor.
    start_time:
        Optional start time in seconds.
    end_time:
        Optional end time in seconds.
    transcription_checkpoint:
        Required checkpoint file path for ``backend="ONSETS_AND_FRAMES"`` or
        ``backend="DEEP_CHROMA"``. No default location is searched. Unused
        for other backends.
    transcription_device:
        Torch device for transcription.
    smoothing_length_seconds:
        Optional temporal smoothing-window duration in seconds; see
        :func:`~tovis.utils.smooth_resample_feature_sequence`.
    output_feature_rate:
        Optional output frame rate in Hz; interpolation is performed on a
        regular time grid at exactly this rate; see
        :func:`~tovis.utils.smooth_resample_feature_sequence`.
    smoothing_window_type:
        Window type passed to
        :func:`~tovis.utils.smooth_resample_feature_sequence`.
    norm:
        Optional frame-wise chroma normalization; see
        :func:`~tovis.utils.normalize_feature_sequence`.
    norm_threshold:
        Threshold passed to :func:`~tovis.utils.normalize_feature_sequence`.

    Returns
    -------
    chromagram, feature_rate:
        Chromagram of shape ``(12, n_frames)`` and its frame rate in Hz.
    """

    backend_normalized = _normalize_backend_name(backend)
    load_offset = 0.0 if start_time is None else start_time
    load_duration = None if end_time is None else end_time - load_offset
    if load_duration is not None and load_duration <= 0:
        raise ValueError("end_time must be greater than start_time")

    if backend_normalized in {"ONSETS_AND_FRAMES", "DEEP_CHROMA"}:
        from .transcription.transcribe import (
            DEEP_CHROMA_SAMPLE_RATE,
            ONSETS_AND_FRAMES_SAMPLE_RATE,
            _validate_checkpoint_path,
        )

        transcription_checkpoint = _validate_checkpoint_path(
            transcription_checkpoint, parameter_name="transcription_checkpoint"
        )
        load_sample_rate = (
            ONSETS_AND_FRAMES_SAMPLE_RATE
            if backend_normalized == "ONSETS_AND_FRAMES"
            else DEEP_CHROMA_SAMPLE_RATE
        )
        audio, actual_sample_rate = librosa.load(
            filename,
            sr=load_sample_rate,
            mono=True,
            offset=load_offset,
            duration=load_duration,
        )
    else:
        audio, actual_sample_rate = librosa.load(
            filename,
            sr=sample_rate,
            mono=True,
            offset=load_offset,
            duration=load_duration,
        )

    return compute_chromagram(
        audio=audio,
        sample_rate=actual_sample_rate,
        backend=backend_normalized,
        window_length=window_length,
        hop_length=hop_length,
        estimate_tuning=estimate_tuning,
        gamma=gamma,
        start_time=None,
        end_time=None,
        transcription_checkpoint=transcription_checkpoint,
        transcription_device=transcription_device,
        smoothing_length_seconds=smoothing_length_seconds,
        output_feature_rate=output_feature_rate,
        smoothing_window_type=smoothing_window_type,
        norm=norm,
        norm_threshold=norm_threshold,
    )


def compute_chromagram_from_annotations(
    note_annotations: Optional[pd.DataFrame] = None,
    feature_rate: Optional[float] = None,
    *,
    filename: Optional[Union[str, Path]] = None,
    start_column: str = "start",
    end_column: str = "end",
    pitch_column: str = "pitch",
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    smoothing_length_seconds: Optional[float] = None,
    output_feature_rate: Optional[float] = None,
    smoothing_window_type: str = "boxcar",
    norm: Optional[NormType] = "2",
    norm_threshold: float = 1e-3,
    kwargs_csv_reader: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, float]:
    """Compute a chromagram from symbolic note annotations.

    Parameters
    ----------
    note_annotations:
        DataFrame containing note events. By default, TOVIS expects
        columns ``start``, ``end``, and ``pitch``.
    feature_rate:
        Output frame rate in Hz.
    filename:
        Optional CSV file to load when ``note_annotations`` is not provided.
        Loaded with :func:`~tovis.utils.read_csv`.
    start_column:
        Column name for note start time.
    end_column:
        Column name for note end time.
    pitch_column:
        Column name for MIDI pitch.
    start_time:
        Optional start time in seconds. If omitted, it is inferred from the
        annotation.
    end_time:
        Optional end time in seconds. If omitted, it is inferred from the
        annotation.
    smoothing_length_seconds:
        Optional temporal smoothing-window duration in seconds. The nearest
        integer number of source feature frames is used; see
        :func:`~tovis.utils.smooth_resample_feature_sequence`.
    output_feature_rate:
        Optional output frame rate in Hz. The chromagram is linearly
        interpolated onto a regular grid at exactly this rate; see
        :func:`~tovis.utils.smooth_resample_feature_sequence`.
    smoothing_window_type:
        Window type passed to
        :func:`~tovis.utils.smooth_resample_feature_sequence`.
    norm:
        Optional frame-wise chroma normalization; see
        :func:`~tovis.utils.normalize_feature_sequence`.
    norm_threshold:
        Threshold passed to :func:`~tovis.utils.normalize_feature_sequence`.
    kwargs_csv_reader:
        Optional keyword arguments passed to :func:`~tovis.utils.read_csv`.

    Returns
    -------
    chromagram, feature_rate:
        Chromagram of shape ``(12, n_frames)`` and its frame rate in Hz.
    """

    if feature_rate is None:
        raise ValueError("feature_rate must be provided")
    if feature_rate <= 0:
        raise ValueError("feature_rate must be positive")
    if note_annotations is None:
        if filename is None:
            raise ValueError("note_annotations or filename must be provided")
        note_annotations = read_csv(filename, **dict(kwargs_csv_reader or {}))

    for column in (start_column, end_column, pitch_column):
        if column not in note_annotations.columns:
            raise ValueError(f'note_annotations must contain a "{column}" column')

    notes = note_annotations[[start_column, end_column, pitch_column]].dropna().astype(float)
    if notes.empty:
        raise ValueError("note_annotations must contain at least one note")

    if start_time is None:
        start_time = float(notes[start_column].min())
    if end_time is None:
        end_time = float(notes[end_column].max())

    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")

    n_frames = int(np.ceil((end_time - start_time) * feature_rate))
    chromagram = np.zeros((12, n_frames), dtype=float)

    for note_start, note_end, pitch in notes.to_numpy():
        clipped_start = max(float(note_start), start_time)
        clipped_end = min(float(note_end), end_time)
        if clipped_end <= clipped_start:
            continue

        pitch_class = int(round(pitch)) % 12
        first_frame = max(0, int(np.floor((clipped_start - start_time) * feature_rate)))
        last_frame = min(n_frames, int(np.ceil((clipped_end - start_time) * feature_rate)))
        chromagram[pitch_class, first_frame:last_frame] += 1.0

    if smoothing_length_seconds is not None or output_feature_rate is not None:
        chromagram, feature_rate = smooth_resample_feature_sequence(
            chromagram,
            feature_rate=feature_rate,
            smoothing_length_seconds=smoothing_length_seconds,
            output_feature_rate=output_feature_rate,
            smoothing_window_type=smoothing_window_type,
        )

    if norm is not None:
        chromagram = normalize_feature_sequence(
            chromagram,
            norm=norm,
            threshold=norm_threshold,
        )

    return chromagram, feature_rate


def _normalize_backend_name(backend: str) -> ChromaBackend:
    """Return the canonical uppercase backend name."""

    backend_upper = backend.upper().replace("-", "_")

    if backend_upper in {"STFT", "CQT", "IIR", "VQT", "ONSETS_AND_FRAMES", "DEEP_CHROMA"}:
        return backend_upper  # type: ignore[return-value]

    raise ValueError(
        'backend must be one of "STFT", "CQT", "IIR", "VQT", '
        '"ONSETS_AND_FRAMES", or "DEEP_CHROMA"'
    )


def _slice_audio(
    audio: np.ndarray,
    *,
    sample_rate: int,
    start_time: Optional[float],
    end_time: Optional[float],
) -> np.ndarray:
    """Return an optional time slice from a mono audio signal."""

    if start_time is None and end_time is None:
        return audio

    start_sample = 0 if start_time is None else int(round(start_time * sample_rate))
    end_sample = audio.size if end_time is None else int(round(end_time * sample_rate))

    if start_sample < 0:
        raise ValueError("start_time must be non-negative")
    if end_sample <= start_sample:
        raise ValueError("end_time must be greater than start_time")

    return audio[start_sample:end_sample]
