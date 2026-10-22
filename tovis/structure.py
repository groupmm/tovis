"""Structure-analysis utilities based on transposition-invariant SSMs."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .utils import normalize_feature_sequence


__all__ = [
    "get_ssm_ti",
    "crop_ssm_ti",
    "get_optimizing_path",
    "get_ssm_index_ti_at_path",
]


def get_ssm_ti(
    feature_sequence: np.ndarray,
    feature_rate: float,
    *,
    reference_sequence: Optional[np.ndarray] = None,
    shifts: Optional[Sequence[int]] = None,
    smoothing_length_seconds: Optional[float] = None,
    tempo_deviation: float = 0.0,
    num_tempo_rel: int = 20,
    direction: str = "both",
    norm: Optional[str] = "2",
    norm_threshold: float = 1e-3,
    thresh: Optional[float] = 0.2,
    threshold_strategy: str = "relative",
    scale: bool = False,
    penalty: float = 0.0,
    min_similarity_for_index: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute a transposition-invariant similarity matrix.

    For feature frames :math:`x_n` and cyclically shifted reference frames
    :math:`y_m^{(s)}`, the unprocessed similarity is
    :math:`S_s(n,m) = x_n^\mathsf{T} y_m^{(s)}`. The output retains
    :math:`\max_s S_s(n,m)` and the maximizing shift at each cell. Optional
    diagonal smoothing allows corresponding passages to differ in local tempo.

    Parameters
    ----------
    feature_sequence:
        Feature matrix of shape ``(n_features, n_frames)``.
    feature_rate:
        Frame rate of ``feature_sequence`` and ``reference_sequence`` in Hz.
    reference_sequence:
        Optional second feature matrix at the same frame rate. If omitted,
        ``feature_sequence`` is compared with itself.
    shifts:
        Cyclic feature-axis shifts (transpositions) to evaluate. If omitted,
        all ``n_features`` shifts are evaluated. For 12-bin chroma features,
        these are the twelve semitone transpositions.
    smoothing_length_seconds:
        Optional duration of diagonal path smoothing in seconds. The nearest
        positive integer number of SSM frames is used.
    tempo_deviation:
        Maximum relative tempo deviation from the main diagonal. For example,
        ``0.3`` allows tempo ratios from ``0.7`` to ``1.3``.
    num_tempo_rel:
        Number of relative-tempo values between the lower and upper deviation
        limits. Ignored when ``tempo_deviation`` is 0.
    direction:
        Diagonal-smoothing direction. Options are ``"forward"``,
        ``"backward"``, and ``"both"``.
    norm:
        Optional frame-wise feature normalization before similarity
        computation. Options are ``"1"``, ``"2"``, ``"max"``, ``"z"``, and
        ``None``; see :func:`~tovis.utils.normalize_feature_sequence`.
    norm_threshold:
        Threshold passed to :func:`~tovis.utils.normalize_feature_sequence`
        for feature normalization.
    thresh:
        Optional similarity threshold. If ``None``, no thresholding is
        applied.
    threshold_strategy:
        Thresholding method. Options are ``"relative"``, which rejects the
        lowest fraction ``thresh`` of matrix cells, and ``"absolute"``, which
        rejects cells satisfying :math:`S < \mathtt{thresh}`.
    scale:
        If true and ``thresh`` is set, map retained values by
        :math:`(S-t)/(S_\mathrm{max}-t)`, where :math:`t` is the effective
        threshold, yielding values in ``[0, 1]``. If ``thresh`` is ``None``,
        positive values are divided by their maximum.
    penalty:
        Value assigned to cells rejected by thresholding.
    min_similarity_for_index:
        If given, transposition indices at cells whose final ``ssm_ti`` value
        is below this value are set to ``-1``.

    Returns
    -------
    ssm_ti, index_ti:
        Transposition-invariant similarity matrix and transposition-index
        matrix, both shaped ``(n_frames, n_reference_frames)``.
    """

    x = _as_feature_matrix(feature_sequence, name="feature_sequence")
    if feature_rate <= 0:
        raise ValueError("feature_rate must be positive")
    y = x if reference_sequence is None else _as_feature_matrix(
        reference_sequence,
        name="reference_sequence",
    )
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature_sequence and reference_sequence must have the same number of rows")

    if norm is not None:
        x = normalize_feature_sequence(x, norm=norm, threshold=norm_threshold)
        y = normalize_feature_sequence(y, norm=norm, threshold=norm_threshold)

    shifts = _prepare_shifts(shifts, n_features=x.shape[0])
    tempo_rel_set = _make_tempo_rel_set(
        tempo_deviation=tempo_deviation,
        num_tempo_rel=num_tempo_rel,
    )
    direction = _normalize_direction(direction)
    smoothing_length_frames = _smoothing_length_to_frames(
        smoothing_length_seconds,
        feature_rate=feature_rate,
    )

    ssm_ti = np.full((x.shape[1], y.shape[1]), -np.inf, dtype=float)
    index_ti = np.full(ssm_ti.shape, -1, dtype=int)

    for shift in shifts:
        y_shifted = np.roll(y, shift=int(shift), axis=0)
        similarity = x.T @ y_shifted
        similarity = _smooth_similarity(
            similarity,
            length_frames=smoothing_length_frames,
            tempo_rel_set=tempo_rel_set,
            direction=direction,
        )
        better = similarity > ssm_ti
        ssm_ti[better] = similarity[better]
        index_ti[better] = int(shift)

    if thresh is not None:
        ssm_ti = _threshold_ssm(
            ssm_ti,
            thresh=thresh,
            strategy=threshold_strategy,
            scale=scale,
            penalty=penalty,
        )
    elif scale:
        ssm_ti = _scale_positive_values(ssm_ti)

    if min_similarity_for_index is not None:
        index_ti[ssm_ti < min_similarity_for_index] = -1

    return ssm_ti, index_ti


def crop_ssm_ti(
    ssm_ti: np.ndarray,
    index_ti: Optional[np.ndarray] = None,
    *,
    feature_rate: float = 1.0,
    x_range: Optional[Sequence[float]] = None,
    y_range: Optional[Sequence[float]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Crop a similarity matrix and, optionally, its index matrix in time.

    Parameters
    ----------
    ssm_ti:
        Similarity matrix of shape ``(n_frames_y, n_frames_x)``.
    index_ti:
        Optional transposition-index matrix with the same shape as ``ssm_ti``.
    feature_rate:
        Frame rate of both SSM axes in Hz. Time limits are mapped to frame
        indices by nearest-integer rounding.
    x_range:
        Optional ``[start, end]`` time interval in seconds for columns. The
        end value is exclusive, as in normal Python slicing.
    y_range:
        Optional ``[start, end]`` time interval in seconds for rows. The end
        value is exclusive, as in normal Python slicing.

    Returns
    -------
    numpy.ndarray or tuple
        Cropped similarity matrix, or a pair ``(ssm_crop, index_crop)`` when
        ``index_ti`` is provided.
    """

    ssm_ti = _as_matrix(ssm_ti, name="ssm_ti")
    if feature_rate <= 0:
        raise ValueError("feature_rate must be positive")
    rows, columns = _crop_slices(
        ssm_ti.shape,
        feature_rate=feature_rate,
        x_range=x_range,
        y_range=y_range,
    )
    ssm_crop = ssm_ti[rows, columns]

    if index_ti is None:
        return ssm_crop

    index_ti = np.asarray(index_ti)
    if index_ti.shape != ssm_ti.shape:
        raise ValueError("index_ti must have the same shape as ssm_ti")
    return ssm_crop, index_ti[rows, columns]


def get_optimizing_path(
    ssm_ti: np.ndarray,
    *,
    method: str = "dtw",
) -> np.ndarray:
    r"""Compute an optimizing path through a similarity matrix.

    Parameters
    ----------
    ssm_ti:
        Similarity matrix.
    method:
        Path objective. ``"dtw"`` computes a full dynamic-time-warping path
        using cost :math:`C=-S`; ``"partial_matching"`` selects an
        order-preserving subset of cell pairs maximizing the sum of their
        similarities, with skipped rows and columns contributing zero.

    Returns
    -------
    numpy.ndarray
        Path as ``(row, column)`` index pairs.
    """

    ssm_ti = _as_matrix(ssm_ti, name="ssm_ti")
    method = method.lower()
    if method == "dtw":
        return _dtw_path(-ssm_ti)
    if method == "partial_matching":
        return _partial_matching_path(ssm_ti)

    raise ValueError('method must be "dtw" or "partial_matching"')


def get_ssm_index_ti_at_path(
    ssm_ti: np.ndarray,
    index_ti: np.ndarray,
    path: np.ndarray,
    *,
    normalize_ssm: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Extract similarity values and transposition indices along a path.

    Parameters
    ----------
    ssm_ti:
        Similarity matrix.
    index_ti:
        Transposition-index matrix with the same shape as ``ssm_ti``.
    path:
        Path as ``(row, column)`` index pairs.
    normalize_ssm:
        If true, transform extracted values by
        :math:`(S-S_\mathrm{min})/(S_\mathrm{max}-S_\mathrm{min})`.
        Constant-valued paths are returned as zeros.

    Returns
    -------
    ssm_path, index_path:
        Similarity values and transposition indices sampled along ``path``.
    """

    ssm_ti = _as_matrix(ssm_ti, name="ssm_ti")
    index_ti = np.asarray(index_ti)
    if index_ti.shape != ssm_ti.shape:
        raise ValueError("index_ti must have the same shape as ssm_ti")

    path = _as_path(path, matrix_shape=ssm_ti.shape)
    ssm_path = ssm_ti[path[:, 0], path[:, 1]].astype(float)
    if normalize_ssm:
        ssm_path = _minmax(ssm_path)

    return ssm_path, index_ti[path[:, 0], path[:, 1]]


def _as_feature_matrix(features: np.ndarray, *, name: str) -> np.ndarray:
    """Return ``features`` as a nonempty two-dimensional float array."""

    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if min(features.shape) == 0:
        raise ValueError(f"{name} must not be empty")
    return features


def _as_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    """Return ``matrix`` as a nonempty two-dimensional float array."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if min(matrix.shape) == 0:
        raise ValueError(f"{name} must not be empty")
    return matrix


def _prepare_shifts(shifts: Optional[Sequence[int]], *, n_features: int) -> np.ndarray:
    """Return a nonempty array of cyclic shifts."""

    if shifts is None:
        return np.arange(n_features, dtype=int)

    shifts = np.asarray(shifts, dtype=int)
    if shifts.size == 0:
        raise ValueError("shifts must contain at least one value")
    return shifts


def _make_tempo_rel_set(*, tempo_deviation: float, num_tempo_rel: int) -> np.ndarray:
    """Return log-spaced tempo ratios in ``[1 - deviation, 1 + deviation]``."""

    tempo_deviation = float(tempo_deviation)
    if tempo_deviation < 0:
        raise ValueError("tempo_deviation must be non-negative")
    if tempo_deviation >= 1:
        raise ValueError("tempo_deviation must be smaller than 1")
    if tempo_deviation == 0:
        return np.array([1.0])

    num_tempo_rel = _validate_positive_int(num_tempo_rel, name="num_tempo_rel")
    lower = 1.0 - tempo_deviation
    upper = 1.0 + tempo_deviation
    return np.exp(np.linspace(np.log(lower), np.log(upper), num_tempo_rel))


def _normalize_direction(direction: str) -> str:
    """Return a validated smoothing direction."""

    direction = direction.lower()
    if direction in {"forward", "backward", "both"}:
        return direction
    raise ValueError('direction must be "forward", "backward", or "both"')


def _validate_positive_int(value: int, *, name: str) -> int:
    """Return ``value`` as a positive integer."""

    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _smoothing_length_to_frames(
    smoothing_length_seconds: Optional[float],
    *,
    feature_rate: float,
) -> int:
    """Convert an optional smoothing duration to the nearest frame count."""

    if smoothing_length_seconds is None:
        return 1
    if smoothing_length_seconds < 0:
        raise ValueError("smoothing_length_seconds must be non-negative")
    return max(1, int(round(smoothing_length_seconds * feature_rate)))


def _threshold_ssm(
    ssm_ti: np.ndarray,
    *,
    thresh: float,
    strategy: str,
    scale: bool,
    penalty: float,
) -> np.ndarray:
    """Threshold an SSM and optionally scale values above its cutoff."""

    if thresh < 0:
        raise ValueError("thresh must be non-negative")

    strategy = strategy.lower()
    if strategy == "absolute":
        rejected = ssm_ti < thresh
        threshold_value = thresh
    elif strategy == "relative":
        if thresh > 1:
            raise ValueError('thresh must be between 0 and 1 for strategy="relative"')
        rejected = np.zeros(ssm_ti.shape, dtype=bool)
        num_cells_to_zero = int(np.floor(thresh * ssm_ti.size))
        if num_cells_to_zero > 0:
            flat_indices = np.argsort(ssm_ti, axis=None)[:num_cells_to_zero]
            rejected.flat[flat_indices] = True
        retained = ssm_ti[~rejected]
        threshold_value = np.min(retained) if retained.size else np.max(ssm_ti)
    else:
        raise ValueError('threshold_strategy must be "relative" or "absolute"')

    thresholded = ssm_ti.copy()
    if scale:
        thresholded = _scale_from_threshold(
            thresholded,
            retained=~rejected,
            threshold_value=threshold_value,
        )
    thresholded[rejected] = penalty
    return thresholded


def _scale_from_threshold(
    values: np.ndarray,
    *,
    retained: np.ndarray,
    threshold_value: float,
) -> np.ndarray:
    """Scale retained values from a threshold value to the maximum onto [0, 1]."""

    scaled = values.copy()
    if not np.any(retained):
        return scaled

    maximum = np.max(values[retained])
    value_range = maximum - threshold_value
    if value_range > 0:
        scaled[retained] = (values[retained] - threshold_value) / value_range
    else:
        scaled[retained] = 1.0
    return scaled


def _scale_positive_values(values: np.ndarray) -> np.ndarray:
    """Scale positive values by their maximum, preserving nonpositive values."""

    scaled = values.copy()
    positive = scaled > 0
    if np.any(positive):
        scaled[positive] = scaled[positive] / np.max(scaled[positive])
    return scaled


def _smooth_similarity(
    similarity: np.ndarray,
    *,
    length_frames: int,
    tempo_rel_set: np.ndarray,
    direction: str,
) -> np.ndarray:
    """Smooth a similarity matrix along diagonal paths."""

    if length_frames == 1 and np.array_equal(tempo_rel_set, np.array([1.0])):
        return similarity

    if direction == "forward":
        return _smooth_similarity_one_direction(
            similarity,
            length_frames=length_frames,
            tempo_rel_set=tempo_rel_set,
            reverse=False,
        )
    if direction == "backward":
        return _smooth_similarity_one_direction(
            similarity,
            length_frames=length_frames,
            tempo_rel_set=tempo_rel_set,
            reverse=True,
        )

    forward = _smooth_similarity_one_direction(
        similarity,
        length_frames=length_frames,
        tempo_rel_set=tempo_rel_set,
        reverse=False,
    )
    backward = _smooth_similarity_one_direction(
        similarity,
        length_frames=length_frames,
        tempo_rel_set=tempo_rel_set,
        reverse=True,
    )
    return np.maximum(forward, backward)


def _smooth_similarity_one_direction(
    similarity: np.ndarray,
    *,
    length_frames: int,
    tempo_rel_set: np.ndarray,
    reverse: bool,
) -> np.ndarray:
    """Smooth a matrix in one diagonal direction over several tempo ratios."""

    smoothed = np.zeros_like(similarity, dtype=float)
    for tempo_rel in tempo_rel_set:
        resampled = _resample_columns(similarity, n_columns=int(np.ceil(similarity.shape[1] / tempo_rel)))
        filtered = _diagonal_mean_filter(resampled, length_frames=length_frames, reverse=reverse)
        restored = _resample_columns(filtered, n_columns=similarity.shape[1])
        smoothed = np.maximum(smoothed, restored)
    return smoothed


def _resample_columns(matrix: np.ndarray, *, n_columns: int) -> np.ndarray:
    """Nearest-neighbor resample matrix columns to ``n_columns``."""

    n_columns = _validate_positive_int(n_columns, name="n_columns")
    source_positions = np.linspace(0, matrix.shape[1] - 1, n_columns)
    source_indices = np.rint(source_positions).astype(int)
    return matrix[:, source_indices]


def _diagonal_mean_filter(matrix: np.ndarray, *, length_frames: int, reverse: bool) -> np.ndarray:
    """Average values over short forward or backward diagonal paths."""

    if reverse:
        return np.flip(
            _diagonal_mean_filter(
                np.flip(matrix),
                length_frames=length_frames,
                reverse=False,
            )
        )

    n_rows, n_columns = matrix.shape
    filtered = np.zeros_like(matrix, dtype=float)
    padded = np.pad(
        matrix,
        ((0, length_frames - 1), (0, length_frames - 1)),
        mode="constant",
    )

    for offset in range(length_frames):
        filtered += padded[offset:offset + n_rows, offset:offset + n_columns]

    return filtered / length_frames


def _crop_slices(
    shape: Tuple[int, int],
    *,
    feature_rate: float,
    x_range: Optional[Sequence[float]],
    y_range: Optional[Sequence[float]],
) -> Tuple[slice, slice]:
    """Convert optional time ranges to matrix slices."""

    n_rows, n_columns = shape
    columns = _range_to_slice(
        x_range,
        upper=n_columns,
        feature_rate=feature_rate,
        name="x_range",
    )
    rows = _range_to_slice(
        y_range,
        upper=n_rows,
        feature_rate=feature_rate,
        name="y_range",
    )
    return rows, columns


def _range_to_slice(
    value_range: Optional[Sequence[float]],
    *,
    upper: int,
    feature_rate: float,
    name: str,
) -> slice:
    """Validate a half-open time range and return its nearest frame slice."""

    if value_range is None:
        return slice(0, upper)
    if len(value_range) != 2:
        raise ValueError(f"{name} must contain exactly two indices")

    start = int(round(float(value_range[0]) * feature_rate))
    end = int(round(float(value_range[1]) * feature_rate))
    if start < 0 or end < start or end > upper:
        duration = upper / feature_rate
        raise ValueError(f"{name} must satisfy 0 <= start <= end <= {duration:g} seconds")
    return slice(start, end)


def _dtw_path(cost: np.ndarray) -> np.ndarray:
    """Return a standard dynamic-time-warping path for a cost matrix."""

    accumulated = _accumulated_cost(cost)
    row, column = np.array(accumulated.shape) - 1
    path = [(row, column)]

    while row > 0 or column > 0:
        if row == 0:
            column -= 1
        elif column == 0:
            row -= 1
        else:
            step_costs = np.array([
                accumulated[row - 1, column - 1],
                accumulated[row - 1, column],
                accumulated[row, column - 1],
            ])
            step = int(np.argmin(step_costs))
            if step == 0:
                row -= 1
                column -= 1
            elif step == 1:
                row -= 1
            else:
                column -= 1
        path.append((row, column))

    path.reverse()
    return np.asarray(path, dtype=int).reshape(-1, 2)


def _accumulated_cost(cost: np.ndarray) -> np.ndarray:
    """Return the accumulated cost matrix for standard DTW."""

    accumulated = np.empty_like(cost, dtype=float)
    accumulated[0, 0] = cost[0, 0]
    accumulated[1:, 0] = cost[1:, 0].cumsum() + accumulated[0, 0]
    accumulated[0, 1:] = cost[0, 1:].cumsum() + accumulated[0, 0]

    for row in range(1, cost.shape[0]):
        for column in range(1, cost.shape[1]):
            accumulated[row, column] = cost[row, column] + min(
                accumulated[row - 1, column - 1],
                accumulated[row - 1, column],
                accumulated[row, column - 1],
            )
    return accumulated


def _partial_matching_path(similarity: np.ndarray) -> np.ndarray:
    """Return a maximum-sum, order-preserving subset of matrix cells."""

    n_rows, n_columns = similarity.shape
    score = np.zeros((n_rows + 1, n_columns + 1), dtype=float)

    for row in range(1, n_rows + 1):
        for column in range(1, n_columns + 1):
            score[row, column] = max(
                score[row - 1, column],
                score[row, column - 1],
                score[row - 1, column - 1] + similarity[row - 1, column - 1],
            )

    path = []
    row, column = n_rows, n_columns
    while row > 0 and column > 0:
        if score[row, column] == score[row - 1, column]:
            row -= 1
        elif score[row, column] == score[row, column - 1]:
            column -= 1
        else:
            path.append((row - 1, column - 1))
            row -= 1
            column -= 1

    path.reverse()
    return np.asarray(path, dtype=int).reshape(-1, 2)


def _as_path(path: np.ndarray, *, matrix_shape: Tuple[int, int]) -> np.ndarray:
    """Validate a path against a matrix shape."""

    path = np.asarray(path, dtype=int)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must have shape (n_steps, 2)")
    if path.size == 0:
        return path

    n_rows, n_columns = matrix_shape
    if np.any(path[:, 0] < 0) or np.any(path[:, 0] >= n_rows):
        raise ValueError("path row indices are out of bounds")
    if np.any(path[:, 1] < 0) or np.any(path[:, 1] >= n_columns):
        raise ValueError("path column indices are out of bounds")
    return path


def _minmax(values: np.ndarray) -> np.ndarray:
    """Return min-max normalized values, preserving constant arrays as zeros."""

    if values.size == 0:
        return values
    value_range = np.max(values) - np.min(values)
    if value_range == 0:
        return np.zeros_like(values)
    return (values - np.min(values)) / value_range
