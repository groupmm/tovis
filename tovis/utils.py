"""Utility helpers for TOVIS."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal


NormType = str


def read_csv(
    filename: Union[str, Path],
    *,
    sep: str = ";",
    **kwargs,
) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame.

    TOVIS annotation files use semicolons by default.

    Parameters
    ----------
    filename:
        Path to the CSV file.
    sep:
        Column separator passed to ``pandas.read_csv``.
    **kwargs:
        Additional keyword arguments passed to ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
        Loaded table.
    """

    return pd.read_csv(filename, sep=sep, **kwargs)


def shorten_annotation(
    annotation: pd.DataFrame,
    *,
    label_column: str = "structure",
) -> pd.DataFrame:
    """Shorten common sonata-form labels in a structure annotation.

    The input DataFrame is not modified. Replacement is substring-based; for
    example, ``"Exposition_FirstGroup"`` becomes ``"E_G1"``.

    Parameters
    ----------
    annotation:
        Structure annotation table.
    label_column:
        Column containing structure labels.

    Returns
    -------
    pandas.DataFrame
        Copy of ``annotation`` with shortened labels.
    """

    if label_column not in annotation.columns:
        raise ValueError(f'annotation must contain a "{label_column}" column')

    annotation_short = annotation.copy()
    replacements = {
        "Exposition": "E",
        "Recapitulation": "R",
        "Coda": "C",
        "FirstGroup": "G1",
        "SecondGroup": "G2",
        "CadentialGroup": "C",
        "Development": "D",
        "Transition": "T",
    }

    labels = annotation_short[label_column].astype(str)
    for long_label, short_label in replacements.items():
        labels = labels.str.replace(long_label, short_label, regex=False)

    annotation_short[label_column] = labels
    return annotation_short


def get_exp_and_repr_limits(
    structure_annotation: pd.DataFrame,
    *,
    start_column: str = "start",
    end_column: str = "end",
    label_column: str = "structure",
) -> Tuple[list[float], list[float]]:
    """Return the first exposition and recapitulation limits.

    Parameters
    ----------
    structure_annotation:
        Structure annotation table.
    start_column:
        Column containing segment start positions.
    end_column:
        Column containing segment end positions.
    label_column:
        Column containing structure labels.

    Returns
    -------
    limits_exp, limits_repr:
        Two ``[start, end]`` lists. A list is empty when the corresponding
        section is not present.
    """

    for column in (start_column, end_column, label_column):
        if column not in structure_annotation.columns:
            raise ValueError(f'structure_annotation must contain a "{column}" column')

    limits_exp = []
    limits_repr = []

    for _, row in structure_annotation.iterrows():
        label = row[label_column]
        if label == "Exposition" and not limits_exp:
            limits_exp = [float(row[start_column]), float(row[end_column])]
        if label == "Recapitulation" and not limits_repr:
            limits_repr = [float(row[start_column]), float(row[end_column])]

    return limits_exp, limits_repr


def key_to_relative_major(
    keys: Union[str, Sequence[str]],
    *,
    suffix_maj: str = "",
    suffix_min: str = "m",
    output: str = "label",
) -> Union[int, str, list[int], list[str]]:
    """Convert key labels to their diatonic center.

    Minor keys are mapped to their relative major. By default, the result is a
    pitch-class label such as ``"C"`` or ``"Ab"``. Use ``output="number"`` for
    numeric pitch classes.

    Parameters
    ----------
    keys:
        Key label or sequence of key labels. Labels may use ``":maj"`` /
        ``":min"`` suffixes or configurable suffixes such as ``"m"`` for
        minor.
    suffix_maj:
        Suffix interpreted as major when labels do not contain a colon.
    suffix_min:
        Suffix interpreted as minor when labels do not contain a colon.
    output:
        ``"label"`` returns flat-spelled key labels; ``"number"`` returns
        pitch classes in ``0..11``.

    Returns
    -------
    str or int or list
        Relative-major labels or pitch-class numbers. The return type follows
        the input shape: a single key returns a single value, a sequence returns
        a list.
    """

    keys_is_sequence = not isinstance(keys, str)
    key_list = list(keys) if keys_is_sequence else [keys]

    diatonic_numbers = [
        _key_label_to_diatonic(key, suffix_maj=suffix_maj, suffix_min=suffix_min)
        for key in key_list
    ]
    if output == "label":
        output_values = [_pitch_class_to_label(number) for number in diatonic_numbers]
    elif output == "number":
        output_values = diatonic_numbers
    else:
        raise ValueError('output must be "label" or "number"')

    if keys_is_sequence:
        return output_values

    return output_values[0]



def key_label_to_pitch_class(label: str) -> int:
    """Convert an ASCII key label to a pitch class.

    Parameters
    ----------
    label:
        Key or pitch-class label such as ``"C"``, ``"F#"``, ``"Ab:maj"``, or
        ``"A:min"``.

    Returns
    -------
    int
        Pitch class in ``0..11`` with C equal to 0.
    """

    return _pitch_class_from_label(label)


def pitch_class_to_key_label(pitch_class: int) -> str:
    """Convert a pitch class to a flat-spelled key label.

    Parameters
    ----------
    pitch_class:
        Pitch class number. Values are interpreted modulo 12.

    Returns
    -------
    str
        Flat-spelled label such as ``"C"``, ``"Db"``, or ``"Ab"``.
    """

    return _pitch_class_to_label(pitch_class)


def normalize_feature_sequence(
    features: np.ndarray,
    *,
    norm: Optional[NormType] = "2",
    threshold: float = 1e-3,
) -> np.ndarray:
    r"""Normalize feature frames with small-frame thresholding.

    For each frame :math:`x`, ``"1"``, ``"2"``, and ``"max"`` compute
    :math:`x/\lVert x\rVert_1`, :math:`x/\lVert x\rVert_2`, and
    :math:`x/\lVert x\rVert_\infty`, respectively. ``"z"`` computes
    :math:`(x-\mu_x)/\sigma_x`.

    Parameters
    ----------
    features:
        Feature matrix with time along the second axis.
    norm:
        Normalization operation. Options are ``"1"``, ``"2"``, ``"max"``,
        ``"z"``, and ``None`` (return the input values unchanged).
    threshold:
        Denominator threshold. Frames satisfying
        :math:`d \leq \mathtt{threshold}` are returned as zero frames, where
        :math:`d` is the selected norm or standard deviation.

    Returns
    -------
    numpy.ndarray
        Frame-normalized feature matrix.
    """

    features = np.asarray(features, dtype=float)

    if norm is None:
        return features
    if norm == "1":
        denominator = np.sum(np.abs(features), axis=0, keepdims=True)
        return _safe_divide(features, denominator, threshold)
    if norm == "2":
        denominator = np.linalg.norm(features, ord=2, axis=0, keepdims=True)
        return _safe_divide(features, denominator, threshold)
    if norm == "max":
        denominator = np.max(np.abs(features), axis=0, keepdims=True)
        return _safe_divide(features, denominator, threshold)
    if norm == "z":
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        return _safe_divide(features - mean, std, threshold)

    raise ValueError('norm must be one of "1", "2", "max", "z", or None')


def smooth_resample_feature_sequence(
    features: np.ndarray,
    feature_rate: float,
    *,
    smoothing_length_seconds: Optional[float] = None,
    output_feature_rate: Optional[float] = None,
    smoothing_window_type: str = "boxcar",
) -> Tuple[np.ndarray, float]:
    """Smooth and resample a feature sequence along its time axis.

    Smoothing is convolution with a unit-sum window. Resampling uses linear
    interpolation on a regular grid with spacing
    ``1 / output_feature_rate`` seconds.

    Parameters
    ----------
    features:
        Feature matrix of shape ``(n_features, n_frames)``.
    feature_rate:
        Input frame rate in Hz.
    smoothing_length_seconds:
        Optional smoothing-window duration in seconds. The nearest positive
        integer number of input frames, obtained by rounding
        ``smoothing_length_seconds * feature_rate``, is used. ``None`` skips
        smoothing.
    output_feature_rate:
        Optional output frame rate in Hz. Values are linearly interpolated onto
        a regular output time grid at exactly this rate. ``None`` preserves the
        input frame grid.
    smoothing_window_type:
        Window specification accepted by ``scipy.signal.get_window``; common
        choices include ``"boxcar"``, ``"hann"``, and ``"hamming"``.

    Returns
    -------
    features_out, feature_rate_out:
        Smoothed and resampled feature matrix and its frame rate in Hz.
    """

    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must have shape (n_features, n_frames)")
    if features.shape[1] == 0:
        raise ValueError("features must contain at least one frame")
    if feature_rate <= 0:
        raise ValueError("feature_rate must be positive")

    features_out = features
    if smoothing_length_seconds is not None:
        if smoothing_length_seconds < 0:
            raise ValueError("smoothing_length_seconds must be non-negative")
        window_length_frames = max(1, int(round(smoothing_length_seconds * feature_rate)))
        filter_kernel = signal.get_window(smoothing_window_type, window_length_frames)
        if np.isclose(np.sum(filter_kernel), 0.0):
            raise ValueError("smoothing window must have a nonzero sum")
        filter_kernel = filter_kernel / np.sum(filter_kernel)
        features_out = signal.convolve(
            features_out,
            filter_kernel[np.newaxis, :],
            mode="same",
        )

    if output_feature_rate is None:
        return features_out, feature_rate
    if output_feature_rate <= 0:
        raise ValueError("output_feature_rate must be positive")

    source_axis = np.arange(features_out.shape[1]) / feature_rate
    n_output_frames = int(np.floor(source_axis[-1] * output_feature_rate)) + 1
    target_axis = np.arange(n_output_frames) / output_feature_rate
    features_out = _resample_features(features_out, source_axis, target_axis, axis=1)
    return features_out, output_feature_rate


def time_to_measures(
    features: Union[pd.DataFrame, np.ndarray, Sequence[float], float],
    sync_annotation: pd.DataFrame,
    feature_rate: Optional[float] = None,
    *,
    columns: Optional[Sequence[str]] = None,
    output_columns: Optional[Sequence[str]] = None,
    inplace: bool = False,
    frames_per_measure: int = 10,
    axis: int = 1,
    time_column: str = "time",
    measure_column: str = "measure",
    time_start: Optional[float] = None,
    start_measure: Optional[float] = None,
    end_measure: Optional[float] = None,
) -> Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Convert seconds to measures or resample time-based features.

    DataFrame inputs are converted column-wise. Scalar and one-dimensional
    inputs are interpreted as time points in
    seconds and are converted directly to measure positions. Two-dimensional
    or higher inputs are interpreted as feature arrays with one temporal axis;
    feature values are then linearly interpolated onto a regular measure grid.

    Parameters
    ----------
    features:
        DataFrame, time point, sequence of time points, or feature array with
        one temporal axis in seconds.
    sync_annotation:
        DataFrame with columns mapping ``time`` in seconds to ``measure``.
    feature_rate:
        Frame rate of the input feature sequence in Hz. Required only when
        ``features`` is a feature array.
    columns:
        DataFrame columns containing time positions in seconds. If omitted for
        DataFrame input, ``time_column`` is converted.
    output_columns:
        Output column names for DataFrame conversion. If omitted, converted
        values overwrite ``columns``.
    inplace:
        If true, modify a DataFrame input in place. Otherwise a copy is
        returned.
    frames_per_measure:
        Number of output frames per measure.
    axis:
        Temporal axis of ``features``. For a standard feature matrix shaped
        ``(n_features, n_frames)``, use ``1``.
    time_column:
        Column name for physical time in ``sync_annotation``.
    measure_column:
        Column name for measure position in ``sync_annotation``.
    time_start:
        Time in seconds of the first feature frame. ``None`` uses zero.
        Ignored for DataFrame, scalar, and one-dimensional inputs.
    start_measure:
        Optional first output measure. If omitted, the feature sequence and
        sync annotation overlap is used.
    end_measure:
        Optional last output measure. If omitted, the feature sequence and
        sync annotation overlap is used.

    Returns
    -------
    pandas.DataFrame or float or numpy.ndarray or tuple
        Converted DataFrame for DataFrame input.
        Converted measure position for scalar input, converted measure
        positions for one-dimensional input, or ``(features_measure,
        measure_axis)`` for feature-array input.
    """

    if frames_per_measure < 1:
        raise ValueError("frames_per_measure must be at least 1")

    sync_times, sync_measures = _prepare_sync_mapping(
        sync_annotation,
        time_column=time_column,
        measure_column=measure_column,
    )

    if isinstance(features, pd.DataFrame):
        return _convert_dataframe_columns(
            features,
            source_axis=sync_times,
            target_axis=sync_measures,
            columns=[time_column] if columns is None else columns,
            output_columns=output_columns,
            inplace=inplace,
        )

    features = np.asarray(features, dtype=float)
    if features.ndim == 0:
        return float(np.interp(float(features), sync_times, sync_measures))
    if features.ndim == 1:
        return np.interp(features, sync_times, sync_measures)

    if feature_rate is None:
        raise ValueError("feature_rate must be provided for feature-array input")
    if feature_rate <= 0:
        raise ValueError("feature_rate must be positive")

    n_frames = features.shape[axis]
    if time_start is None:
        time_start = 0.0
    source_times = time_start + np.arange(n_frames) / feature_rate

    if start_measure is None or end_measure is None:
        start_time = max(float(source_times[0]), float(sync_times[0]))
        end_time = min(float(source_times[-1]), float(sync_times[-1]))
        if end_time < start_time:
            raise ValueError("feature time axis does not overlap sync annotation")
        if start_measure is None:
            start_measure = float(np.interp(start_time, sync_times, sync_measures))
        if end_measure is None:
            end_measure = float(np.interp(end_time, sync_times, sync_measures))

    measure_axis = _regular_axis(start_measure, end_measure, 1.0 / frames_per_measure)
    target_times = np.interp(measure_axis, sync_measures, sync_times)
    features_measure = _resample_features(features, source_times, target_times, axis=axis)

    return features_measure, measure_axis


def measures_to_time(
    features: Union[pd.DataFrame, np.ndarray, Sequence[float], float],
    sync_annotation: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    output_columns: Optional[Sequence[str]] = None,
    inplace: bool = False,
    frames_per_measure: int = 10,
    feature_rate: Optional[float] = None,
    axis: int = 1,
    time_column: str = "time",
    measure_column: str = "measure",
    measure_start: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Convert measures to seconds or resample measure-based features.

    DataFrame inputs are converted column-wise. Scalar and one-dimensional
    inputs are interpreted as measure positions and
    are converted directly to seconds. Two-dimensional or higher inputs are
    interpreted as feature arrays with one temporal axis in measures.
    ``measure_start`` should then match the first measure position of the
    feature array. If omitted, the first measure in ``sync_annotation`` is
    used. Both the measure/time map and feature values are interpolated
    linearly.

    Parameters
    ----------
    features:
        DataFrame, measure position, sequence of measure positions, or feature
        array with one temporal axis in measures.
    sync_annotation:
        DataFrame mapping measure positions to physical time in seconds.
    columns:
        DataFrame columns containing measure positions. If omitted for
        DataFrame input, ``measure_column`` is converted.
    output_columns:
        Output column names for DataFrame conversion. If omitted, converted
        values overwrite ``columns``.
    inplace:
        If true, modify a DataFrame input in place. Otherwise a copy is
        returned.
    frames_per_measure:
        Number of input frames per measure.
    feature_rate:
        Output frame rate in Hz. If omitted, a rate is inferred from the
        median local tempo.
    axis:
        Temporal axis of ``features``. For a standard feature matrix shaped
        ``(n_features, n_frames)``, use ``1``.
    time_column:
        Column name for physical time in ``sync_annotation``.
    measure_column:
        Column name for measure position in ``sync_annotation``.
    measure_start:
        Measure position of the first input frame.
    start_time:
        Optional first output time in seconds.
    end_time:
        Optional last output time in seconds.

    Returns
    -------
    pandas.DataFrame or float or numpy.ndarray or tuple
        Converted DataFrame for DataFrame input.
        Converted time in seconds for scalar input, converted times for
        one-dimensional input, or ``(features_time, time_axis)`` for
        feature-array input.
    """

    if frames_per_measure < 1:
        raise ValueError("frames_per_measure must be at least 1")

    sync_times, sync_measures = _prepare_sync_mapping(
        sync_annotation,
        time_column=time_column,
        measure_column=measure_column,
    )

    if isinstance(features, pd.DataFrame):
        return _convert_dataframe_columns(
            features,
            source_axis=sync_measures,
            target_axis=sync_times,
            columns=[measure_column] if columns is None else columns,
            output_columns=output_columns,
            inplace=inplace,
        )

    features = np.asarray(features, dtype=float)
    if features.ndim == 0:
        return float(np.interp(float(features), sync_measures, sync_times))
    if features.ndim == 1:
        return np.interp(features, sync_measures, sync_times)

    n_frames = features.shape[axis]
    if measure_start is None:
        measure_start = float(sync_measures[0])

    source_measures = measure_start + np.arange(n_frames) / frames_per_measure

    if feature_rate is None:
        seconds_per_measure = np.diff(sync_times) / np.diff(sync_measures)
        feature_rate = frames_per_measure / float(np.median(seconds_per_measure))
    if feature_rate <= 0:
        raise ValueError("feature_rate must be positive")

    if start_time is None or end_time is None:
        start_measure = max(float(source_measures[0]), float(sync_measures[0]))
        end_measure = min(float(source_measures[-1]), float(sync_measures[-1]))
        if end_measure < start_measure:
            raise ValueError("feature measure axis does not overlap sync annotation")
        if start_time is None:
            start_time = float(np.interp(start_measure, sync_measures, sync_times))
        if end_time is None:
            end_time = float(np.interp(end_measure, sync_measures, sync_times))

    time_axis = _regular_axis(start_time, end_time, 1.0 / feature_rate)
    target_measures = np.interp(time_axis, sync_times, sync_measures)
    features_time = _resample_features(features, source_measures, target_measures, axis=axis)

    return features_time, time_axis


def get_tempo_curve(
    measure_annotation: pd.DataFrame,
    *,
    time_column: str = "time",
    measure_column: str = "measure",
    smoothing_length_measures: Optional[float] = None,
    smoothing_method: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute a local tempo curve from time/measure annotations.

    For consecutive annotation points, tempo is
    :math:`(m_{i+1}-m_i)/(t_{i+1}-t_i)` measures per second and is assigned to
    the interval midpoint. The midpoint axis is returned in seconds and
    measures.

    Parameters
    ----------
    measure_annotation:
        Annotation table containing time/measure mapping points.
    time_column:
        Column name for physical time.
    measure_column:
        Column name for measure position.
    smoothing_length_measures:
        Optional smoothing-window length in measures. For each tempo sample at
        measure position :math:`m_i`, samples within
        ``smoothing_length_measures / 2`` measures are aggregated. ``None``
        leaves the curve unsmoothed.
    smoothing_method:
        Aggregation method for smoothing. Options are ``"mean"`` and
        ``"median"``.

    Returns
    -------
    axis_seconds, axis_measures, tempo:
        Midpoint axes in seconds and measures, plus local tempo in measures per
        second.
    """

    times, measures = _prepare_measure_annotation(
        measure_annotation,
        time_column=time_column,
        measure_column=measure_column,
    )

    delta_time = np.diff(times)
    delta_measure = np.diff(measures)
    valid = delta_time > 0
    if not np.any(valid):
        raise ValueError("measure_annotation must contain increasing time values")

    tempo = delta_measure[valid] / delta_time[valid]
    axis_seconds = (times[:-1][valid] + times[1:][valid]) / 2
    axis_measures = (measures[:-1][valid] + measures[1:][valid]) / 2

    if smoothing_length_measures is not None:
        tempo = _smooth_tempo_curve_by_measure(
            tempo,
            axis_measures,
            smoothing_length_measures=smoothing_length_measures,
            smoothing_method=smoothing_method,
        )

    return axis_seconds, axis_measures, tempo


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Divide frames safely, returning zero where denominators are too small."""

    output = np.zeros_like(numerator, dtype=float)
    valid = np.squeeze(denominator, axis=0) > threshold

    if np.any(valid):
        output[:, valid] = numerator[:, valid] / denominator[:, valid]

    return output


def _key_label_to_diatonic(
    key: str,
    *,
    suffix_maj: str,
    suffix_min: str,
) -> int:
    """Convert a key label to its relative-major pitch class."""

    root_pitch_class, mode = _parse_key_label(
        key,
        suffix_maj=suffix_maj,
        suffix_min=suffix_min,
    )
    if mode == "minor":
        return (root_pitch_class + 3) % 12

    return root_pitch_class % 12


def _pitch_class_from_label(label: str) -> int:
    """Parse a key or pitch label and return its root pitch class."""

    pitch_class, _ = _parse_key_label(label, suffix_maj="", suffix_min="m")
    return pitch_class


def _parse_key_label(
    label: str,
    *,
    suffix_maj: str = "",
    suffix_min: str = "m",
) -> Tuple[int, Optional[str]]:
    """Parse an ASCII key label into root pitch class and mode."""

    label = str(label).strip()
    label = label.replace("-", "b")
    mode = None

    if ":" in label:
        root, mode_label = label.split(":", maxsplit=1)
        mode_label = mode_label.strip().lower()
        if mode_label in {"maj", "major", ""}:
            mode = "major"
        elif mode_label in {"min", "minor", "m"}:
            mode = "minor"
        else:
            raise ValueError(f'Unsupported key mode "{mode_label}" in key label "{label}"')
    elif suffix_min and label.endswith(suffix_min):
        root = label[: -len(suffix_min)]
        mode = "minor"
    elif suffix_maj and label.endswith(suffix_maj):
        root = label[: -len(suffix_maj)]
        mode = "major"
    else:
        root = label
        mode = "major"

    root = root.strip().lower()

    if not root:
        raise ValueError("key label must contain a pitch name")

    pitch_classes = {
        "c": 0,
        "d": 2,
        "e": 4,
        "f": 5,
        "g": 7,
        "a": 9,
        "b": 11,
    }
    if root[0] not in pitch_classes:
        raise ValueError(f'Unsupported key label "{label}"')

    pitch_class = pitch_classes[root[0]]
    for accidental in root[1:]:
        if accidental == "#":
            pitch_class += 1
        elif accidental == "b":
            pitch_class -= 1
        else:
            raise ValueError(f'Unsupported accidental "{accidental}" in key label "{label}"')

    return pitch_class % 12, mode


def _pitch_class_to_label(pitch_class: int) -> str:
    """Return the flat-spelled label for a pitch class."""

    labels = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
    return labels[pitch_class % 12]


def _prepare_measure_annotation(
    measure_annotation: pd.DataFrame,
    *,
    time_column: str,
    measure_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and extract sorted measure-annotation arrays."""

    if time_column not in measure_annotation.columns:
        raise ValueError(f'measure_annotation must contain a "{time_column}" column')
    if measure_column not in measure_annotation.columns:
        raise ValueError(f'measure_annotation must contain a "{measure_column}" column')

    mapping = measure_annotation[[time_column, measure_column]].dropna().astype(float)
    mapping = mapping.sort_values(time_column)
    mapping = mapping.drop_duplicates(subset=time_column, keep="first")

    times = mapping[time_column].to_numpy()
    measures = mapping[measure_column].to_numpy()
    if times.size < 2:
        raise ValueError("measure_annotation must contain at least two points")

    return times, measures


def _smooth_tempo_curve_by_measure(
    tempo: np.ndarray,
    axis_measures: np.ndarray,
    *,
    smoothing_length_measures: float,
    smoothing_method: str,
) -> np.ndarray:
    """Smooth tempo values over local measure-domain neighborhoods."""

    if smoothing_length_measures < 0:
        raise ValueError("smoothing_length_measures must be non-negative")
    if smoothing_method not in {"mean", "median"}:
        raise ValueError('smoothing_method must be "mean" or "median"')
    if smoothing_length_measures == 0 or tempo.size <= 1:
        return tempo

    half_length = smoothing_length_measures / 2
    tempo_smoothed = np.empty_like(tempo, dtype=float)

    for index, center_measure in enumerate(axis_measures):
        in_window = np.abs(axis_measures - center_measure) <= half_length
        values = tempo[in_window]
        if smoothing_method == "mean":
            tempo_smoothed[index] = np.mean(values)
        else:
            tempo_smoothed[index] = np.median(values)

    return tempo_smoothed


def _prepare_sync_mapping(
    sync_annotation: pd.DataFrame,
    *,
    time_column: str,
    measure_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and extract a strictly increasing sync mapping."""

    if time_column not in sync_annotation.columns:
        raise ValueError(f'sync_annotation must contain a "{time_column}" column')
    if measure_column not in sync_annotation.columns:
        raise ValueError(f'sync_annotation must contain a "{measure_column}" column')

    mapping = sync_annotation[[time_column, measure_column]].dropna().astype(float)
    mapping = mapping.sort_values(time_column)
    mapping = mapping.drop_duplicates(subset=time_column, keep="first")
    mapping = mapping.drop_duplicates(subset=measure_column, keep="first")

    sync_times = mapping[time_column].to_numpy()
    sync_measures = mapping[measure_column].to_numpy()

    if sync_times.size < 2:
        raise ValueError("sync_annotation must contain at least two mapping points")
    if np.any(np.diff(sync_times) <= 0):
        raise ValueError("sync_annotation time values must be strictly increasing")
    if np.any(np.diff(sync_measures) <= 0):
        raise ValueError("sync_annotation measure values must be strictly increasing")

    return sync_times, sync_measures


def _regular_axis(start: float, end: float, step: float) -> np.ndarray:
    """Create a regular axis that includes the end point within tolerance."""

    if step <= 0:
        raise ValueError("step must be positive")
    if end < start:
        raise ValueError("axis end must be greater than or equal to axis start")

    return np.arange(start, end + step * 0.5, step)


def _convert_dataframe_columns(
    dataframe: pd.DataFrame,
    *,
    source_axis: np.ndarray,
    target_axis: np.ndarray,
    columns: Sequence[str],
    output_columns: Optional[Sequence[str]],
    inplace: bool,
) -> pd.DataFrame:
    """Interpolate selected DataFrame columns from one time axis to another."""

    if isinstance(columns, str):
        columns = [columns]
    columns = list(columns)
    if not columns:
        raise ValueError("columns must contain at least one column name")

    if output_columns is None:
        output_columns = columns
    elif isinstance(output_columns, str):
        output_columns = [output_columns]
    else:
        output_columns = list(output_columns)

    if len(output_columns) != len(columns):
        raise ValueError("output_columns must match the number of columns")

    missing_columns = [column for column in columns if column not in dataframe.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    dataframe_out = dataframe if inplace else dataframe.copy()
    for source_column, target_column in zip(columns, output_columns):
        values = pd.to_numeric(dataframe_out[source_column], errors="coerce").to_numpy()
        converted = np.full(values.shape, np.nan, dtype=float)
        valid = np.isfinite(values)
        converted[valid] = np.interp(values[valid], source_axis, target_axis)
        dataframe_out[target_column] = converted

    return dataframe_out


def _resample_features(
    features: np.ndarray,
    source_axis: np.ndarray,
    target_axis: np.ndarray,
    *,
    axis: int,
) -> np.ndarray:
    """Linearly resample an array along an arbitrary axis."""

    moved = np.moveaxis(features, axis, -1)
    original_shape = moved.shape
    flattened = moved.reshape(-1, original_shape[-1])
    resampled = np.empty((flattened.shape[0], target_axis.size), dtype=float)

    for row_index, row in enumerate(flattened):
        resampled[row_index] = np.interp(target_axis, source_axis, row)

    resampled = resampled.reshape(*original_shape[:-1], target_axis.size)
    return np.moveaxis(resampled, -1, axis)
