"""Visualization utilities for TOVIS."""

from __future__ import annotations

import warnings
from itertools import permutations
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

from .scale import get_scale_labels
from .utils import (
    get_tempo_curve,
    key_label_to_pitch_class,
    key_to_relative_major,
    read_csv,
    shorten_annotation as shorten_structure_annotation,
)


CHROMA_LABELS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
TRANSPOSITION_INDEX_LABELS = tuple(str(index) for index in range(12))


def get_transposition_index_colors() -> list[str]:
    """Return colors for inactive cells and the 12 transposition indices.

    Returns
    -------
    list of str
        Thirteen colors. The first color corresponds to index ``-1`` and the
        remaining colors correspond to transposition indices ``0`` to ``11``.
    """

    return [
        "white",
        "gray",
        "green",
        "darkblue",
        "pink",
        "yellow",
        "magenta",
        "cyan",
        "red",
        "orange",
        "purple",
        "lightsteelblue",
        "royalblue",
    ]


def get_transposition_index_cmap() -> mcolors.ListedColormap:
    """Return a discrete colormap for transposition-index plots."""

    return mcolors.ListedColormap(
        get_transposition_index_colors(),
        name="tovis-transposition-index",
    )


def get_major_minor_contrast_cmap() -> mcolors.LinearSegmentedColormap:
    """Return a red-black-blue colormap for major/minor contrast plots."""

    return mcolors.LinearSegmentedColormap.from_list(
        "tovis-major-minor-contrast",
        ["red", "black", "blue"],
    )


def transposition_index_to_rgba(
    index_ti: np.ndarray,
    ssm_ti: np.ndarray,
    *,
    alpha_min: float = 0.0,
) -> np.ndarray:
    """Convert transposition indices and similarities to an RGBA image.

    The RGB channels encode the transposition index. The alpha channel encodes
    the similarity value, normalized continuously between the finite minimum
    and maximum of nonnegative similarities.

    Parameters
    ----------
    index_ti:
        Transposition-index matrix with values from ``-1`` to ``11``.
    ssm_ti:
        Similarity matrix with the same shape as ``index_ti``.
    alpha_min:
        Minimum alpha value assigned to finite nonnegative similarity values.
        Use a small value such as ``0.15`` when very weak cells should remain
        faintly visible.

    Returns
    -------
    numpy.ndarray
        RGBA image of shape ``(n_rows, n_columns, 4)``.
    """

    index_ti = _validate_transposition_index_matrix(index_ti)
    ssm_ti = np.asarray(ssm_ti, dtype=float)
    if ssm_ti.shape != index_ti.shape:
        raise ValueError("ssm_ti must have the same shape as index_ti")

    colors = np.asarray([
        mcolors.to_rgba(color)
        for color in get_transposition_index_colors()
    ])
    rgba = colors[index_ti.astype(int) + 1]
    rgba[:, :, 3] = _similarity_to_alpha(ssm_ti, alpha_min=alpha_min)
    return rgba


def overlay_measure_annotations(
    measure_annotation: Optional[pd.DataFrame] = None,
    *,
    filename: Optional[Union[str, Path]] = None,
    measure_annotation_hop: int = 10,
    ax: Optional[plt.Axes] = None,
    direction: str = "vertical",
    labels: bool = True,
    label_location: str = "bottom",
    time_column: str = "time",
    measure_column: str = "measure",
    kwargs_csv_reader: Optional[Mapping[str, Any]] = None,
    kwargs_lines: Optional[Mapping[str, Any]] = None,
    kwargs_labels: Optional[Mapping[str, Any]] = None,
) -> plt.Axes:
    """Overlay measure positions on an existing axis.

    The function draws one line per selected measure annotation and optionally
    labels it with the measure number. Existing x- and y-axis limits are
    restored after drawing, so overlays respect the visible region of the
    underlying plot.

    Parameters
    ----------
    measure_annotation:
        Annotation table containing measure positions. By default, TOVIS
        expects columns ``time`` and ``measure``.
    filename:
        Optional CSV file to load when ``measure_annotation`` is not provided.
        Loaded with :func:`~tovis.utils.read_csv`.
    measure_annotation_hop:
        Plot every n-th measure annotation. Must be at least 1.
    ax:
        Matplotlib axis on which the overlay is drawn.
    direction:
        ``"vertical"`` draws time positions along the x-axis; ``"horizontal"``
        draws time positions along the y-axis.
    labels:
        If true, draw measure-number labels in addition to the lines.
    label_location:
        Label anchor within the axis, either ``"bottom"`` or ``"top"``.
    time_column:
        Column name for the physical time values.
    measure_column:
        Column name for measure numbers.
    kwargs_csv_reader:
        Optional keyword arguments passed to :func:`~tovis.utils.read_csv`.
    kwargs_lines:
        Optional keyword arguments passed to Matplotlib line drawing.
    kwargs_labels:
        Optional keyword arguments passed to Matplotlib text drawing.

    Returns
    -------
    matplotlib.axes.Axes
        The input axis with the measure overlay added.
    """

    if ax is None:
        raise ValueError("ax must be passed to function")
    if measure_annotation is None:
        if filename is None:
            raise ValueError("measure_annotation or filename must be provided")
        measure_annotation = read_csv(filename, **dict(kwargs_csv_reader or {}))
    if measure_annotation_hop < 1:
        raise ValueError("measure_annotation_hop must be at least 1")

    line_kwargs = {
        "color": "tab:red",
        "linestyle": "--",
        "linewidth": 1,
    }
    line_kwargs.update(dict(kwargs_lines or {}))

    label_kwargs = {
        "color": "tab:red",
        "fontsize": 8,
    }
    label_kwargs.update(_annotation_label_alignment(label_location, direction))
    label_kwargs.update(dict(kwargs_labels or {}))

    label_axis_position = _annotation_label_axis_position(label_location)
    times = _get_annotation_values(measure_annotation, time_column, 0)
    measures = _get_annotation_values(measure_annotation, measure_column, 1)

    measure_times = []
    measure_labels = []
    for time, measure in zip(times, measures):
        measure_float = float(measure)
        if np.isclose(measure_float % measure_annotation_hop, 0.0):
            measure_times.append(float(time))
            measure_labels.append(_format_measure_label(measure_float))

    xlims_original = ax.get_xlim()
    ylims_original = ax.get_ylim()

    if direction == "vertical":
        measure_times, measure_labels = _filter_positions_by_limits(
            measure_times,
            measure_labels,
            xlims_original,
        )
        ax.vlines(measure_times, ymin=ylims_original[0], ymax=ylims_original[1], **line_kwargs)
        if labels:
            for measure_time, measure_label in zip(measure_times, measure_labels):
                ax.text(
                    x=measure_time,
                    y=label_axis_position,
                    s=" " + measure_label,
                    transform=ax.get_xaxis_transform(),
                    clip_on=True,
                    **label_kwargs,
                )
    elif direction == "horizontal":
        measure_times, measure_labels = _filter_positions_by_limits(
            measure_times,
            measure_labels,
            ylims_original,
        )
        ax.hlines(measure_times, xmin=xlims_original[0], xmax=xlims_original[1], **line_kwargs)
        if labels:
            for measure_time, measure_label in zip(measure_times, measure_labels):
                ax.text(
                    x=label_axis_position,
                    y=measure_time,
                    s=" " + measure_label,
                    rotation=90,
                    transform=ax.get_yaxis_transform(),
                    clip_on=True,
                    **label_kwargs,
                )
    else:
        raise ValueError('direction must be "vertical" or "horizontal"')

    ax.set_xlim(xlims_original)
    ax.set_ylim(ylims_original)

    return ax


def overlay_structure_annotations(
    structure_annotation: Optional[pd.DataFrame] = None,
    *,
    filename: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
    direction: str = "vertical",
    labels: bool = True,
    label_location: str = "top",
    shorten_annotation: bool = False,
    start_column: str = "start",
    label_column: str = "structure",
    kwargs_csv_reader: Optional[Mapping[str, Any]] = None,
    kwargs_lines: Optional[Mapping[str, Any]] = None,
    kwargs_labels: Optional[Mapping[str, Any]] = None,
) -> plt.Axes:
    """Overlay formal-structure boundaries on an existing axis.

    The function draws structure-start boundaries and optionally labels them
    with the corresponding structure names. Existing axis limits are restored
    after drawing.

    Parameters
    ----------
    structure_annotation:
        Annotation table. By default, TOVIS expects ``start`` and
        ``structure`` columns.
    filename:
        Optional CSV file to load when ``structure_annotation`` is not
        provided. Loaded with :func:`~tovis.utils.read_csv`.
    ax:
        Matplotlib axis on which the overlay is drawn.
    direction:
        ``"vertical"`` draws boundary positions along the x-axis;
        ``"horizontal"`` draws boundary positions along the y-axis.
    labels:
        If true, draw structure labels in addition to the boundary lines.
    label_location:
        Label anchor within the axis, either ``"bottom"`` or ``"top"``.
    shorten_annotation:
        If true, shorten common sonata-form labels using
        :func:`~tovis.utils.shorten_annotation` before plotting.
    start_column:
        Column name for boundary positions.
    label_column:
        Column name for structure labels.
    kwargs_csv_reader:
        Optional keyword arguments passed to :func:`~tovis.utils.read_csv`.
    kwargs_lines:
        Optional keyword arguments passed to Matplotlib line drawing.
    kwargs_labels:
        Optional keyword arguments passed to Matplotlib text drawing.

    Returns
    -------
    matplotlib.axes.Axes
        The input axis with the structure overlay added.
    """

    if ax is None:
        raise ValueError("ax must be passed to function")
    if structure_annotation is None:
        if filename is None:
            raise ValueError("structure_annotation or filename must be provided")
        structure_annotation = read_csv(filename, **dict(kwargs_csv_reader or {}))
    if shorten_annotation:
        structure_annotation = shorten_structure_annotation(
            structure_annotation,
            label_column=label_column,
        )

    line_kwargs = {
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 2,
    }
    line_kwargs.update(dict(kwargs_lines or {}))

    label_kwargs = {
        "color": "tab:blue",
        "fontsize": 10,
    }
    label_kwargs.update(_annotation_label_alignment(label_location, direction))
    label_kwargs.update(dict(kwargs_labels or {}))

    label_axis_position = _annotation_label_axis_position(label_location)
    structure_starts = [
        float(structure_start)
        for structure_start in _get_annotation_values(structure_annotation, start_column, 0)
    ]
    structure_labels = [
        str(structure_label)
        for structure_label in _get_annotation_values(structure_annotation, label_column, 2)
    ]

    xlims_original = ax.get_xlim()
    ylims_original = ax.get_ylim()

    if direction == "vertical":
        structure_starts, structure_labels = _filter_positions_by_limits(
            structure_starts,
            structure_labels,
            xlims_original,
        )
        ax.vlines(structure_starts, ymin=ylims_original[0], ymax=ylims_original[1], **line_kwargs)
        if labels:
            for structure_start, structure_label in zip(structure_starts, structure_labels):
                ax.text(
                    x=structure_start,
                    y=label_axis_position,
                    s=" " + structure_label,
                    transform=ax.get_xaxis_transform(),
                    clip_on=True,
                    **label_kwargs,
                )
    elif direction == "horizontal":
        structure_starts, structure_labels = _filter_positions_by_limits(
            structure_starts,
            structure_labels,
            ylims_original,
        )
        ax.hlines(structure_starts, xmin=xlims_original[0], xmax=xlims_original[1], **line_kwargs)
        if labels:
            for structure_start, structure_label in zip(structure_starts, structure_labels):
                ax.text(
                    x=label_axis_position,
                    y=structure_start,
                    s=" " + structure_label,
                    rotation=90,
                    transform=ax.get_yaxis_transform(),
                    clip_on=True,
                    **label_kwargs,
                )
    else:
        raise ValueError('direction must be "vertical" or "horizontal"')

    ax.set_xlim(xlims_original)
    ax.set_ylim(ylims_original)

    return ax


def overlay_localkey_annotations(
    localkey_annotation: Optional[pd.DataFrame] = None,
    *,
    filename: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
    direction: str = "horizontal",
    start_column: str = "start",
    end_column: str = "end",
    localkey_column: str = "localkey",
    scale_label_version: str = "absolute_diatonic",
    sort_by_fifths: bool = False,
    global_key: Optional[str] = None,
    fill: bool = False,
    major_color: str = "blue",
    minor_color: str = "red",
    kwargs_csv_reader: Optional[Mapping[str, Any]] = None,
    kwargs_boxes: Optional[Mapping[str, Any]] = None,
) -> plt.Axes:
    """Overlay local-key annotations as boxes on a scale-likelihood plot.

    Each annotation row is drawn as an unfilled rectangle spanning
    ``start`` to ``end`` in time and one scale row in the plot. Major keys are
    framed in blue by default and minor keys in red. Minor keys are placed on
    the row of their relative major, matching the diatonic scale-likelihood
    representation where, for example, C major and A minor share one row.

    Parameters
    ----------
    localkey_annotation:
        Annotation table. By default, TOVIS expects ``start``, ``end``, and
        ``localkey`` columns.
    filename:
        Optional CSV file to load when ``localkey_annotation`` is not
        provided. Loaded with :func:`~tovis.utils.read_csv`.
    ax:
        Matplotlib scale-likelihood axis on which the boxes are drawn.
    direction:
        Direction of the scale-likelihood plot, either ``"horizontal"`` or
        ``"vertical"``.
    start_column:
        Column name for annotation start positions in the plot's time unit.
    end_column:
        Column name for annotation end positions in the plot's time unit.
    localkey_column:
        Column name for local-key labels such as ``"C:maj"``, ``"A:min"``,
        ``"C"``, or ``"Am"``.
    scale_label_version:
        Label version used by the underlying scale-likelihood plot. Options
        are ``"absolute_diatonic"`` and ``"relative_diatonic"``.
    sort_by_fifths:
        If true, place boxes on rows sorted by circle-of-fifths order, matching
        ``plot_scale_likelihood(sort_by_fifths=True)``.
    global_key:
        Optional global key used when ``scale_label_version`` is
        ``"relative_diatonic"`` or when fifths sorting is centered.
    fill:
        If true, fill boxes with a transparent version of the major/minor
        color. If false, draw frames only.
    major_color:
        Box edge color for major local keys.
    minor_color:
        Box edge color for minor local keys.
    kwargs_csv_reader:
        Optional keyword arguments passed to :func:`~tovis.utils.read_csv`.
    kwargs_boxes:
        Optional keyword arguments passed to
        :class:`matplotlib.patches.Rectangle`.

    Returns
    -------
    matplotlib.axes.Axes
        The input axis with local-key boxes added.
    """

    if ax is None:
        raise ValueError("ax must be passed to function")
    if direction not in {"horizontal", "vertical"}:
        raise ValueError('direction must be "horizontal" or "vertical"')
    if localkey_annotation is None:
        if filename is None:
            raise ValueError("localkey_annotation or filename must be provided")
        localkey_annotation = read_csv(filename, **dict(kwargs_csv_reader or {}))

    required_columns = {start_column, end_column, localkey_column}
    missing_columns = required_columns.difference(localkey_annotation.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"localkey_annotation is missing required columns: {missing}")

    scale_labels = get_scale_labels(
        scale_label_version,
        global_key="C" if global_key is None else global_key,
    )
    row_lookup = _get_scale_row_lookup(
        scale_labels,
        sort_by_fifths=sort_by_fifths,
        global_key=global_key,
    )

    box_kwargs = {
        "linewidth": 2,
        "clip_on": True,
    }
    box_kwargs.update(dict(kwargs_boxes or {}))

    xlims_original = ax.get_xlim()
    ylims_original = ax.get_ylim()

    for _, row in localkey_annotation.iterrows():
        if pd.isna(row[start_column]) or pd.isna(row[end_column]) or pd.isna(row[localkey_column]):
            continue

        start = float(row[start_column])
        end = float(row[end_column])
        if not np.isfinite(start) or not np.isfinite(end) or np.isclose(start, end):
            continue
        start, end = sorted((start, end))

        localkey = str(row[localkey_column]).strip()
        try:
            row_position = row_lookup[int(key_to_relative_major(localkey, output="number"))]
        except (KeyError, ValueError):
            continue

        mode = _get_localkey_mode(localkey)
        edgecolor = minor_color if mode == "minor" else major_color
        if direction == "horizontal":
            xy = (start, row_position - 0.5)
            width = end - start
            height = 1.0
        else:
            xy = (row_position - 0.5, start)
            width = 1.0
            height = end - start

        rectangle_kwargs = dict(box_kwargs)
        rectangle_kwargs.setdefault("edgecolor", edgecolor)
        if fill:
            rectangle_kwargs.setdefault("facecolor", mcolors.to_rgba(edgecolor, 0.18))
        else:
            rectangle_kwargs.setdefault("facecolor", "none")

        ax.add_patch(
            mpatches.Rectangle(
                xy,
                width,
                height,
                **rectangle_kwargs,
            )
        )

    ax.set_xlim(xlims_original)
    ax.set_ylim(ylims_original)
    return ax


def make_diamond_marker(
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    *,
    marker_slope: float = 0.5,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create polygon coordinates for a slanted diamond correspondence marker.

    The marker connects two time intervals, one on the x-axis and one on the
    y-axis of an SSM-style plot. Its main direction follows the line from
    ``(x_vals[0], y_vals[0])`` to ``(x_vals[1], y_vals[1])``; ``marker_slope``
    controls how wide the marker opens around that direction.

    Parameters
    ----------
    x_vals:
        Two x-axis positions, usually the start and end time of one formal
        part.
    y_vals:
        Two y-axis positions, usually the start and end time of a
        corresponding formal part.
    marker_slope:
        Relative opening of the diamond. Larger values produce a wider marker.
    n_points:
        Number of interpolation points per diamond side.

    Returns
    -------
    x_marker, y_marker:
        Coordinates that can be passed to ``matplotlib.axes.Axes.plot``.
    """

    if len(x_vals) != 2 or len(y_vals) != 2:
        raise ValueError("x_vals and y_vals must each contain two values")
    if marker_slope <= 0:
        raise ValueError("marker_slope must be positive")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    x1, x2 = np.asarray(x_vals, dtype=float)
    y1, y2 = np.asarray(y_vals, dtype=float)
    if not np.all(np.isfinite([x1, x2, y1, y2])):
        raise ValueError("marker coordinates must be finite")
    if np.isclose(x1, x2):
        raise ValueError("x_vals must span a nonzero interval")

    x_times = np.linspace(x1, x2, n_points)
    slope = (y2 - y1) / (x2 - x1)
    slope_steep = slope * (1 + marker_slope)
    slope_flat = slope / (1 + marker_slope)

    upper_1 = slope_steep * x_times + (y1 - slope_steep * x1)
    upper_2 = slope_flat * x_times + (y2 - slope_flat * x2)
    lower_1 = slope_flat * x_times + (y1 - slope_flat * x1)
    lower_2 = slope_steep * x_times + (y2 - slope_steep * x2)

    y_upper = np.minimum(upper_1, upper_2)
    y_lower = np.maximum(lower_1, lower_2)
    return (
        np.concatenate([x_times, x_times[::-1]]),
        np.concatenate([y_upper, y_lower[::-1]]),
    )


def overlay_correspondence_annotations(
    ann_parts: Union[pd.DataFrame, str, Path],
    ann_correspondences: Union[pd.DataFrame, str, Path],
    *,
    ax: Optional[plt.Axes] = None,
    marker_slope: float = 0.5,
    start_column: str = "start",
    end_column: str = "end",
    harmony_column: str = "harmony",
    name_column: str = "name",
    modulation_label: str = "Mod",
    kwargs_csv_reader: Optional[Mapping[str, Any]] = None,
    kwargs_lines: Optional[Mapping[str, Any]] = None,
) -> plt.Axes:
    """Overlay structural correspondences on an SSM-TI axis.

    ``ann_parts`` describes formal parts with start/end coordinates, a harmony
    label, and a unique part name. ``ann_correspondences`` describes groups of
    corresponding parts: each row lists part names that should be connected
    pairwise. Each directed pair is drawn as a slanted diamond on the SSM
    axis. The coordinate unit is not changed by this function; use
    :func:`~tovis.utils.time_to_measures` or
    :func:`~tovis.utils.measures_to_time` beforehand if the annotation table
    needs conversion.

    The marker color encodes the transposition from the first part's harmony
    to the second part's harmony using :func:`get_transposition_index_colors`.
    If either harmony is equal to ``modulation_label`` or cannot be parsed, the
    marker is drawn in black.

    Parameters
    ----------
    ann_parts:
        DataFrame or CSV filename with part annotations.
    ann_correspondences:
        DataFrame or CSV filename whose rows list corresponding part names.
    ax:
        Matplotlib SSM axis on which the correspondence markers are drawn.
    marker_slope:
        Relative opening of each diamond marker.
    start_column:
        Column in ``ann_parts`` containing part starts in the SSM axis unit.
    end_column:
        Column in ``ann_parts`` containing part ends in the SSM axis unit.
    harmony_column:
        Column in ``ann_parts`` containing harmony or key labels.
    name_column:
        Column in ``ann_parts`` containing unique part names.
    modulation_label:
        Harmony label used for modulatory parts. Pairs containing this label
        are drawn in black.
    kwargs_csv_reader:
        Optional keyword arguments passed to :func:`~tovis.utils.read_csv`.
    kwargs_lines:
        Optional keyword arguments passed to Matplotlib line drawing.

    Returns
    -------
    matplotlib.axes.Axes
        The input axis with correspondence annotations added.
    """

    if ax is None:
        raise ValueError("ax must be passed to function")

    csv_kwargs = dict(kwargs_csv_reader or {})
    if not isinstance(ann_parts, pd.DataFrame):
        ann_parts = read_csv(ann_parts, **csv_kwargs)
    if not isinstance(ann_correspondences, pd.DataFrame):
        ann_correspondences = read_csv(ann_correspondences, **csv_kwargs)

    required_columns = {start_column, end_column, harmony_column, name_column}
    missing_columns = required_columns.difference(ann_parts.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"ann_parts is missing required columns: {missing}")

    line_kwargs = {
        "linewidth": 2,
    }
    line_kwargs.update(dict(kwargs_lines or {}))

    transposition_colors = np.asarray(get_transposition_index_colors()[1:])
    part_lookup = {
        _normalize_correspondence_part_name(row[name_column]): row
        for _, row in ann_parts.iterrows()
        if not pd.isna(row[name_column])
    }

    xlims_original = ax.get_xlim()
    ylims_original = ax.get_ylim()

    for _, row in ann_correspondences.iterrows():
        part_names = _get_correspondence_part_names(row)
        for part_x, part_y in permutations(part_names, 2):
            if part_x not in part_lookup or part_y not in part_lookup:
                continue

            row_x = part_lookup[part_x]
            row_y = part_lookup[part_y]

            try:
                marker_x, marker_y = make_diamond_marker(
                    [row_x[start_column], row_x[end_column]],
                    [row_y[start_column], row_y[end_column]],
                    marker_slope=marker_slope,
                )
            except ValueError:
                continue

            color = _get_correspondence_color(
                row_x[harmony_column],
                row_y[harmony_column],
                colors=transposition_colors,
                modulation_label=modulation_label,
            )
            ax.plot(marker_x, marker_y, color=color, **line_kwargs)

    ax.set_xlim(xlims_original)
    ax.set_ylim(ylims_original)
    return ax


def overlay_lines(
    ax: plt.Axes,
    *,
    direction: str = "horizontal",
    positions: Optional[Sequence[float]] = None,
    spacing: float = 1.0,
    color: str = "lightgray",
    linewidth: float = 0.8,
    linestyle: str = "-",
    alpha: float = 0.7,
    zorder: float = 0.0,
) -> plt.Axes:
    """Overlay reference lines on an existing axis.

    By default, lines are drawn at unit-spaced positions halfway between
    integer tick locations, for example ``-0.5, 0.5, 1.5, ...``. Axis limits
    are preserved.

    Parameters
    ----------
    ax:
        Matplotlib axis on which the lines are drawn.
    direction:
        Line orientation. ``"horizontal"`` draws lines at y-axis positions;
        ``"vertical"`` draws lines at x-axis positions.
    positions:
        Optional axis positions for the lines. If omitted, positions are
        generated from ``spacing``.
    spacing:
        Distance between generated lines when ``positions`` is omitted.
    color:
        Line color.
    linewidth:
        Line width.
    linestyle:
        Matplotlib line style.
    alpha:
        Line opacity.
    zorder:
        Drawing order of the lines.

    Returns
    -------
    matplotlib.axes.Axes
        The input axis with reference lines added.
    """

    if direction not in {"horizontal", "vertical"}:
        raise ValueError('direction must be "horizontal" or "vertical"')

    xlims_original = ax.get_xlim()
    ylims_original = ax.get_ylim()
    position_limits = ylims_original if direction == "horizontal" else xlims_original

    if positions is None:
        if spacing <= 0:
            raise ValueError("spacing must be positive")
        lower, upper = sorted(position_limits)
        start = np.floor(lower / spacing) * spacing - spacing / 2
        stop = upper + spacing
        positions = np.arange(start, stop, spacing)

    positions = [
        float(position)
        for position in positions
        if np.isfinite(position)
    ]
    lower, upper = sorted(position_limits)
    for position in positions:
        if lower <= position <= upper:
            line_function = ax.axhline if direction == "horizontal" else ax.axvline
            line_function(
                position,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder,
            )

    ax.set_xlim(xlims_original)
    ax.set_ylim(ylims_original)
    return ax


def plot_matrix(
    matrix: np.ndarray,
    *,
    direction: str = "horizontal",
    feature_rate: float = 1.0,
    offset: float = 0.0,
    frequency_rate: float = 1.0,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    clim: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
    xlabel: str = "Time (seconds)",
    ylabel: str = "",
    xlabels: Optional[Sequence[str]] = None,
    ylabels: Optional[Sequence[str]] = None,
    ytick_label_hop: int = 1,
    force_ytick_labels: Optional[Sequence[str]] = None,
    colorbar: bool = True,
    colorbar_label: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 3),
    cmap: str = "gray_r",
    interpolation: str = "None",
    aspect: str = "auto",
) -> Tuple[plt.Figure, plt.Axes, plt.AxesImage]:
    """Plot a feature matrix as an image.

    By default, the first matrix axis is shown vertically and the second axis
    horizontally. With ``direction="vertical"``, the plot is rotated so that
    time runs vertically. The time axis is interpreted in seconds using
    ``feature_rate``.

    Parameters
    ----------
    matrix:
        Matrix to display, usually shaped ``(n_features, n_frames)``.
    direction:
        Time-axis direction, either ``"horizontal"`` or ``"vertical"``.
    feature_rate:
        Number of matrix columns per second.
    offset:
        Time value of the first matrix frame in seconds.
    frequency_rate:
        Number of matrix rows per vertical-axis unit when ``ylabels`` is not
        provided.
    xlim:
        Optional visible horizontal-axis limits.
    ylim:
        Optional visible vertical-axis limits.
    clim:
        Optional color limits passed to the image.
    title:
        Axis title.
    xlabel:
        Horizontal-axis label.
    ylabel:
        Vertical-axis label.
    xlabels:
        Optional labels for discrete x-axis positions.
    ylabels:
        Optional labels for matrix rows.
    ytick_label_hop:
        Show only every n-th y-axis label. Must be at least 1.
    force_ytick_labels:
        Labels that should be shown even if skipped by ``ytick_label_hop``.
    colorbar:
        If true, add a colorbar to the figure.
    colorbar_label:
        Label for the colorbar.
    ax:
        Existing axis to draw into. If omitted, a new figure and axis are
        created.
    figsize:
        Figure size used when creating a new figure.
    cmap:
        Colormap passed to ``matplotlib.axes.Axes.imshow``.
    interpolation:
        Image interpolation passed to ``matplotlib.axes.Axes.imshow``.
    aspect:
        Image aspect mode passed to ``matplotlib.axes.Axes.imshow``.
    
    Returns
    -------
    fig, ax, image:
        The Matplotlib figure, axis, and image artist.
    """

    if direction not in {"horizontal", "vertical"}:
        raise ValueError('direction must be "horizontal" or "vertical"')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    matrix = np.asarray(matrix)
    if ylabels is None:
        feature_min = 0
        feature_max = matrix.shape[0] / frequency_rate
    else:
        feature_min = -0.5
        feature_max = len(ylabels) - 0.5

    time_min = offset
    time_max = offset + matrix.shape[1] / feature_rate
    if direction == "horizontal":
        matrix_display = matrix
        extent = [time_min, time_max, feature_min, feature_max]
        origin = "lower"
    else:
        matrix_display = np.rot90(matrix, k=-1)
        extent = [feature_min, feature_max, time_min, time_max]
        origin = "lower"

    image = ax.imshow(
        matrix_display,
        aspect=aspect,
        origin=origin,
        interpolation=interpolation,
        extent=extent,
        cmap=cmap,
    )

    if clim is not None:
        image.set_clim(clim)
    if direction == "horizontal":
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    else:
        if xlim is not None:
            ax.set_ylim(xlim)
        if ylim is not None:
            ax.set_xlim(ylim)
    if title is not None:
        ax.set_title(title)

    if direction == "horizontal":
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)

    if ylabels is not None:
        tick_positions, tick_labels = _get_hopped_tick_labels(
            ylabels,
            hop=ytick_label_hop,
            force_labels=force_ytick_labels,
        )
        if direction == "horizontal":
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
            ax.set_ylim(feature_min, feature_max)
        else:
            tick_positions_rotated = feature_min + feature_max - tick_positions
            ax.set_xticks(tick_positions_rotated)
            ax.set_xticklabels(tick_labels)
            ax.set_xlim(feature_min, feature_max)

    if xlabels is not None:
        x_tick_positions = offset + (np.arange(len(xlabels)) + 0.5) / feature_rate
        if direction == "horizontal":
            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels(xlabels)
        else:
            ax.set_yticks(x_tick_positions)
            ax.set_yticklabels(xlabels)

    if colorbar:
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(colorbar_label)

    return fig, ax, image


def plot_cmy_overlay(
    features: Sequence[np.ndarray],
    labels: Optional[Sequence[str]] = None,
    *,
    clip: bool = True,
    legend: bool = True,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes, plt.AxesImage]:
    """Plot two or three feature matrices as a CMY-style overlay.

    For two inputs, high values are shown as cyan and magenta, with overlap in
    blue. For three inputs, high values are shown as cyan, magenta, and yellow.
    Matrices may have different numbers of frames; shorter matrices are
    zero-padded to the length of the longest input.

    Parameters
    ----------
    features:
        Two or three matrices with the same number of rows.
    labels:
        Optional names used for the legend entries.
    clip:
        If true, clip feature values to the interval ``[0, 1]`` before color
        mixing.
    legend:
        If true and ``labels`` are provided, add a legend describing the color
        mixtures.
    **kwargs:
        Additional keyword arguments forwarded to ``plot_matrix``.

    Returns
    -------
    fig, ax, image:
        The Matplotlib figure, axis, and image artist.
    """

    if len(features) not in {2, 3}:
        raise ValueError("features must contain two or three feature matrices")

    feature_arrays = [np.asarray(feature, dtype=float) for feature in features]
    if any(feature.ndim != 2 for feature in feature_arrays):
        raise ValueError("all feature matrices must be two-dimensional")
    n_rows = feature_arrays[0].shape[0]
    if any(feature.shape[0] != n_rows for feature in feature_arrays):
        raise ValueError("all feature matrices must have the same number of rows")

    if labels is not None and len(labels) != len(features):
        raise ValueError("labels must match the number of feature matrices")

    feature_arrays = _pad_feature_matrices_to_longest(feature_arrays)
    feature_shape = feature_arrays[0].shape
    features_cmy = np.stack(feature_arrays, axis=2)
    if len(feature_arrays) == 2:
        features_cmy = np.concatenate(
            [features_cmy, np.zeros((*feature_shape, 1), dtype=float)],
            axis=2,
        )

    if clip:
        features_cmy = np.clip(features_cmy, 0.0, 1.0)
    features_rgb = 1.0 - features_cmy

    fig, ax, image = plot_matrix(features_rgb, colorbar=False, **kwargs)

    if legend and labels is not None:
        legend_handles = _get_cmy_legend_handles(labels)
        ax.legend(handles=legend_handles, loc="lower right", ncols=2, fontsize="x-small")

    return fig, ax, image


def plot_transposition_index_matrix(
    index_ti: np.ndarray,
    *,
    ssm_ti: Optional[np.ndarray] = None,
    alpha_min: float = 0.0,
    feature_rate: float = 1.0,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    title: Optional[str] = "Transposition index matrix",
    xlabel: str = "Time (seconds)",
    ylabel: str = "Time (seconds)",
    colorbar: bool = True,
    colorbar_label: str = "Transposition index",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 7),
    interpolation: str = "none",
) -> Tuple[plt.Figure, plt.Axes, plt.AxesImage]:
    """Plot a transposition-index matrix.

    The matrix is expected to contain integer transposition indices ``-1`` to
    ``11``. Cells with value ``-1`` are treated as inactive and shown with the
    first color of the transposition-index colormap. If ``ssm_ti`` is provided,
    transposition indices are encoded as RGB colors and similarity values are
    encoded continuously as alpha values.

    Parameters
    ----------
    index_ti:
        Transposition-index matrix of shape ``(n_rows, n_columns)``.
    ssm_ti:
        Optional similarity matrix with the same shape as ``index_ti``. When
        provided, its values control the alpha channel of the image.
    alpha_min:
        Minimum alpha value for finite nonnegative similarity values in RGBA
        mode.
    feature_rate:
        Matrix frame rate in Hz for both axes.
    xlim:
        Optional visible horizontal time limits in seconds.
    ylim:
        Optional visible vertical time limits in seconds.
    title:
        Plot title.
    xlabel:
        Horizontal-axis label.
    ylabel:
        Vertical-axis label.
    colorbar:
        If true, add a discrete colorbar.
    colorbar_label:
        Label for the colorbar.
    ax:
        Existing axis to draw into. If omitted, a new figure and axis are
        created.
    figsize:
        Figure size used when creating a new figure.
    interpolation:
        Interpolation mode passed to ``imshow``.

    Returns
    -------
    fig, ax, image:
        The Matplotlib figure, axis, and image artist. The plotted matrix uses
        an equal x/y data aspect ratio.
    """

    index_ti = _validate_transposition_index_matrix(index_ti)

    cmap = get_transposition_index_cmap()
    norm = _get_transposition_index_norm()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    extent = [
        0,
        index_ti.shape[1] / feature_rate,
        0,
        index_ti.shape[0] / feature_rate,
    ]
    if ssm_ti is None:
        image = ax.imshow(
            index_ti,
            aspect="equal",
            origin="lower",
            interpolation=interpolation,
            extent=extent,
            cmap=cmap,
            norm=norm,
        )
        colorbar_mappable = image
    else:
        rgba = transposition_index_to_rgba(index_ti, ssm_ti, alpha_min=alpha_min)
        image = ax.imshow(
            rgba,
            aspect="equal",
            origin="lower",
            interpolation=interpolation,
            extent=extent,
        )
        colorbar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar_mappable.set_array([])

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if colorbar:
        colorbar_artist = fig.colorbar(colorbar_mappable, ax=ax, ticks=np.arange(-1, 12))
        colorbar_artist.set_label(colorbar_label)
        colorbar_artist.set_ticklabels([""] + list(TRANSPOSITION_INDEX_LABELS))
        image.colorbar = colorbar_artist

    return fig, ax, image


def plot_features_and_ssm(
    index_ti: np.ndarray,
    *,
    ssm_ti: Optional[np.ndarray] = None,
    features_x: Optional[np.ndarray] = None,
    features_y: Optional[np.ndarray] = None,
    feature_rate_ssm: float = 1.0,
    feature_rate_x: Optional[float] = None,
    feature_rate_y: Optional[float] = None,
    ssm_offset_x: float = 0.0,
    ssm_offset_y: float = 0.0,
    features_offset_x: float = 0.0,
    features_offset_y: float = 0.0,
    feature_labels_x: Optional[Sequence[str]] = None,
    feature_labels_y: Optional[Sequence[str]] = None,
    alpha_min: float = 0.0,
    title: Optional[str] = "Transposition index matrix",
    xlabel: str = "Time X (seconds)",
    ylabel: str = "Time Y (seconds)",
    feature_ylabel_x: str = "",
    feature_xlabel_y: str = "",
    colorbar: bool = True,
    colorbar_label: str = "Transposition index",
    figsize: Tuple[float, float] = (10, 9),
    feature_panel_ratio: float = 0.3,
    corner_text: Optional[str] = None,
    interpolation: str = "none",
    kwargs_features_x: Optional[Mapping[str, Any]] = None,
    kwargs_features_y: Optional[Mapping[str, Any]] = None,
    kwargs_corner_text: Optional[Mapping[str, Any]] = None,
) -> Tuple[
    plt.Figure,
    Tuple[plt.Axes, plt.Axes, plt.Axes],
    Tuple[plt.AxesImage, Optional[plt.AxesImage], Optional[plt.AxesImage]],
]:
    """Plot an SSM-TI with aligned feature matrices below and to its left.

    The main upper-right axis shows the transposition-index matrix, optionally
    weighted by ``ssm_ti``. ``features_x`` is plotted below and shares the
    SSM's horizontal time axis. ``features_y`` is rotated by 90 degrees,
    plotted on the left, and shares the SSM's vertical time axis. The SSM uses
    equal data scaling and its panel proportions follow its temporal shape.

    Parameters
    ----------
    index_ti:
        Transposition-index matrix shaped ``(n_frames_y, n_frames_x)``.
    ssm_ti:
        Optional similarity matrix with the same shape as ``index_ti``. When
        provided, it controls the SSM-TI alpha channel.
    features_x:
        Optional feature matrix corresponding to SSM columns, shaped
        ``(n_features_x, n_frames_x_features)``.
    features_y:
        Optional feature matrix corresponding to SSM rows, shaped
        ``(n_features_y, n_frames_y_features)``.
    feature_rate_ssm:
        Frame rate of both SSM axes in Hz.
    feature_rate_x:
        Frame rate of ``features_x`` in Hz. Defaults to ``feature_rate_ssm``.
    feature_rate_y:
        Frame rate of ``features_y`` in Hz. Defaults to ``feature_rate_ssm``.
    ssm_offset_x:
        Time offset of the SSM x-axis in seconds.
    ssm_offset_y:
        Time offset of the SSM y-axis in seconds.
    features_offset_x:
        Time offset of ``features_x`` in seconds.
    features_offset_y:
        Time offset of ``features_y`` in seconds.
    feature_labels_x:
        Optional row labels for the bottom feature matrix.
    feature_labels_y:
        Optional feature labels for the rotated left feature matrix.
    alpha_min:
        Minimum alpha value passed to
        :func:`plot_transposition_index_matrix`.
    title:
        Title for the SSM-TI axis.
    xlabel:
        Horizontal time-axis label.
    ylabel:
        Vertical time-axis label.
    feature_ylabel_x:
        Vertical-axis label for the bottom feature matrix.
    feature_xlabel_y:
        Horizontal-axis label for the rotated left feature matrix.
    colorbar:
        If true, add a discrete transposition-index colorbar.
    colorbar_label:
        Label for the transposition-index colorbar.
    figsize:
        Figure size.
    feature_panel_ratio:
        Size of each feature panel relative to the adjacent SSM dimension.
    corner_text:
        Optional text displayed in the empty lower-left panel.
    interpolation:
        Interpolation mode used for the SSM-TI.
    kwargs_features_x:
        Optional keyword arguments passed to :func:`plot_matrix` for
        ``features_x``.
    kwargs_features_y:
        Optional keyword arguments passed to :func:`plot_matrix` for the
        rotated ``features_y``.
    kwargs_corner_text:
        Optional keyword arguments passed to the lower-left text artist.

    Returns
    -------
    fig, axes, images:
        Figure; tuple containing the SSM, bottom-feature, and left-feature
        axes; and their image artists. A missing feature matrix returns
        ``None`` for its image artist while leaving its axis available.
    """

    index_ti = _validate_transposition_index_matrix(index_ti)
    if feature_rate_ssm <= 0:
        raise ValueError("feature_rate_ssm must be positive")
    feature_rate_x = feature_rate_ssm if feature_rate_x is None else feature_rate_x
    feature_rate_y = feature_rate_ssm if feature_rate_y is None else feature_rate_y
    if feature_rate_x <= 0 or feature_rate_y <= 0:
        raise ValueError("feature rates must be positive")
    if feature_panel_ratio <= 0:
        raise ValueError("feature_panel_ratio must be positive")
    offsets = (ssm_offset_x, ssm_offset_y, features_offset_x, features_offset_y)
    if not all(np.isfinite(offset) for offset in offsets):
        raise ValueError("all offsets must be finite")

    if ssm_ti is not None:
        ssm_ti = np.asarray(ssm_ti, dtype=float)
        if ssm_ti.shape != index_ti.shape:
            raise ValueError("ssm_ti must have the same shape as index_ti")
    if features_x is not None:
        features_x = _validate_alignment_feature_matrix(features_x, name="features_x")
    if features_y is not None:
        features_y = _validate_alignment_feature_matrix(features_y, name="features_y")

    duration_x = index_ti.shape[1] / feature_rate_ssm
    duration_y = index_ti.shape[0] / feature_rate_ssm
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(feature_panel_ratio * duration_x, duration_x),
        height_ratios=(duration_y, feature_panel_ratio * duration_y),
        wspace=0.05,
        hspace=0.05,
    )
    ax_ssm = fig.add_subplot(grid[0, 1])
    ax_features_x = fig.add_subplot(grid[1, 1], sharex=ax_ssm)
    ax_features_y = fig.add_subplot(grid[0, 0], sharey=ax_ssm)
    ax_empty = fig.add_subplot(grid[1, 0])
    ax_empty.set_axis_off()
    if corner_text is not None:
        text_kwargs = {
            "horizontalalignment": "center",
            "verticalalignment": "center",
            "wrap": True,
        }
        text_kwargs.update(dict(kwargs_corner_text or {}))
        ax_empty.text(0.5, 0.5, corner_text, transform=ax_empty.transAxes, **text_kwargs)

    _, _, image_ssm = plot_transposition_index_matrix(
        index_ti,
        ssm_ti=ssm_ti,
        alpha_min=alpha_min,
        feature_rate=feature_rate_ssm,
        title=title,
        xlabel="",
        ylabel="",
        colorbar=False,
        ax=ax_ssm,
        interpolation=interpolation,
    )
    image_ssm.set_extent(
        [
            ssm_offset_x,
            ssm_offset_x + duration_x,
            ssm_offset_y,
            ssm_offset_y + duration_y,
        ]
    )
    ax_ssm.set_xlim(ssm_offset_x, ssm_offset_x + duration_x)
    ax_ssm.set_ylim(ssm_offset_y, ssm_offset_y + duration_y)
    ax_ssm.set_aspect("equal", adjustable="box")
    ax_ssm.tick_params(axis="x", labelbottom=False)
    ax_ssm.tick_params(axis="y", labelleft=False)

    image_x = _plot_aligned_feature_x(
        features_x,
        ax=ax_features_x,
        duration=duration_x,
        feature_rate=feature_rate_x,
        offset=features_offset_x,
        labels=feature_labels_x,
        xlabel=xlabel,
        ylabel=feature_ylabel_x,
        kwargs=kwargs_features_x,
    )
    image_y = _plot_aligned_feature_y(
        features_y,
        ax=ax_features_y,
        duration=duration_y,
        feature_rate=feature_rate_y,
        offset=features_offset_y,
        labels=feature_labels_y,
        xlabel=feature_xlabel_y,
        ylabel=ylabel,
        kwargs=kwargs_features_y,
    )

    ax_ssm.set_xlim(ssm_offset_x, ssm_offset_x + duration_x)
    ax_ssm.set_ylim(ssm_offset_y, ssm_offset_y + duration_y)
    ax_ssm.apply_aspect()
    _align_feature_axes_to_ssm(
        ax_ssm,
        ax_features_x=ax_features_x,
        ax_features_y=ax_features_y,
        ax_corner=ax_empty,
    )

    if colorbar:
        colorbar_mappable = plt.cm.ScalarMappable(
            norm=_get_transposition_index_norm(),
            cmap=get_transposition_index_cmap(),
        )
        colorbar_mappable.set_array([])
        ssm_position = ax_ssm.get_position()
        colorbar_axis = fig.add_axes(
            [
                ssm_position.x1 + 0.01,
                ssm_position.y0,
                0.02,
                ssm_position.height,
            ]
        )
        colorbar_artist = fig.colorbar(
            colorbar_mappable,
            cax=colorbar_axis,
            ticks=np.arange(-1, 12),
        )
        colorbar_artist.set_label(colorbar_label)
        colorbar_artist.set_ticklabels([""] + list(TRANSPOSITION_INDEX_LABELS))

    return (
        fig,
        (ax_ssm, ax_features_x, ax_features_y),
        (image_ssm, image_x, image_y),
    )


def plot_transposition_index(
    index_ti_path: np.ndarray,
    path: np.ndarray,
    *,
    ssm_path: Optional[np.ndarray] = None,
    feature_rate: float = 1.0,
    offset_time: float = 0.0,
    sort_by_fifths: bool = False,
    weighted_scatter: bool = True,
    marker_size: float = 50.0,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 3),
    title: Optional[str] = None,
    xlabel: str = "Time (seconds)",
    ylabel: str = "Transposition index",
    colorbar: bool = True,
    colorbar_label: str = "Transposition index",
    grid: bool = True,
) -> Tuple[plt.Figure, plt.Axes, plt.Collection]:
    """Plot transposition indices sampled along an optimizing path.

    Parameters
    ----------
    index_ti_path:
        Transposition-index values along ``path``. Entries with value ``-1``
        are inactive and omitted from the plot.
    path:
        Optimizing path as ``(row, column)`` index pairs. The first path column
        is used as the time axis.
    ssm_path:
        Optional similarity values along the path. If ``weighted_scatter`` is
        true, these values control marker size.
    feature_rate:
        Path-index frame rate in Hz.
    offset_time:
        Constant offset added to the time axis in seconds.
    sort_by_fifths:
        If true, arrange transposition indices by circle-of-fifths distance
        and place index ``0`` at the vertical center. Y-axis labels show
        relative diatonic steps from ``-6`` to ``+5``, and the y-axis label is
        set to ``"Rel. diatonic transposition"``.
    weighted_scatter:
        If true, scale marker areas by ``ssm_path``.
    marker_size:
        Maximum marker size.
    ax:
        Existing axis to draw into. If omitted, a new figure and axis are
        created.
    figsize:
        Figure size used when creating a new figure.
    title:
        Plot title.
    xlabel:
        Horizontal-axis label.
    ylabel:
        Vertical-axis label.
    colorbar:
        If true, add a discrete colorbar.
    colorbar_label:
        Label for the colorbar.
    grid:
        If true, draw a grid.

    Returns
    -------
    fig, ax, scatter:
        The Matplotlib figure, axis, and scatter artist.
    """

    index_ti_path = np.asarray(index_ti_path)
    path = np.asarray(path)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must have shape (n_steps, 2)")
    if index_ti_path.shape[0] != path.shape[0]:
        raise ValueError("index_ti_path and path must have the same length")
    if np.any(index_ti_path < -1) or np.any(index_ti_path > 11):
        raise ValueError("index_ti_path values must be between -1 and 11")
    if not np.all(np.equal(index_ti_path, np.floor(index_ti_path))):
        raise ValueError("index_ti_path must contain integer values")

    active = index_ti_path != -1
    index_ti_path = index_ti_path[active].astype(int)
    path = path[active]
    if ssm_path is not None:
        ssm_path = np.asarray(ssm_path)
        if ssm_path.ndim != 1 or ssm_path.shape[0] != active.shape[0]:
            raise ValueError("ssm_path must be one-dimensional and match path length")
        ssm_path = ssm_path[active]

    if weighted_scatter:
        if ssm_path is None:
            raise ValueError("ssm_path must be provided when weighted_scatter is True")
        scatter_weights = _normalize_marker_weights(ssm_path)
    else:
        scatter_weights = np.ones_like(index_ti_path, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    time_axis = path[:, 0] / feature_rate + offset_time
    if sort_by_fifths:
        y_values, ytick_labels = _sort_transposition_indices_by_fifths(index_ti_path)
        ylabel = "Rel. diatonic transposition"
    else:
        y_values = index_ti_path
        ytick_labels = list(TRANSPOSITION_INDEX_LABELS)

    scatter = ax.scatter(
        x=time_axis,
        y=y_values,
        c=index_ti_path,
        s=scatter_weights * marker_size,
        cmap=get_transposition_index_cmap(),
        norm=_get_transposition_index_norm(),
    )

    ax.set_ylim(-0.5, 11.5)
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(ytick_labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)

    if colorbar:
        colorbar_artist = fig.colorbar(scatter, ax=ax, ticks=np.arange(-1, 12))
        colorbar_artist.set_label(colorbar_label)
        colorbar_artist.set_ticklabels([""] + list(TRANSPOSITION_INDEX_LABELS))

    return fig, ax, scatter


def plot_features_and_alignment(
    features_x: Optional[np.ndarray] = None,
    features_y: Optional[np.ndarray] = None,
    path: Optional[np.ndarray] = None,
    ssm_path: Optional[np.ndarray] = None,
    index_ti_path: Optional[np.ndarray] = None,
    *,
    feature_rate_x: float = 1.0,
    feature_rate_y: float = 1.0,
    feature_rate_path: float = 1.0,
    duration_x: Optional[float] = None,
    duration_y: Optional[float] = None,
    feature_labels_x: Optional[Sequence[str]] = None,
    feature_labels_y: Optional[Sequence[str]] = None,
    title_x: Optional[str] = "Sequence X",
    title_y: Optional[str] = "Sequence Y",
    xlabel_x: str = "Time (seconds)",
    xlabel_y: str = "Time (seconds)",
    ylabel_x: str = "",
    ylabel_y: str = "",
    alignment_hop: int = 1,
    ssm_controls_alpha: bool = False,
    ssm_controls_linewidth: bool = True,
    linewidth_min: float = 0.25,
    linewidth_max: float = 3.0,
    alignment_alpha: float = 0.8,
    colorbar: bool = True,
    colorbar_label: str = "Transposition index",
    figsize: Tuple[float, float] = (10, 7),
    height_ratios: Sequence[float] = (2.0, 1.0, 2.0),
    kwargs_features_x: Optional[Mapping[str, Any]] = None,
    kwargs_features_y: Optional[Mapping[str, Any]] = None,
) -> Tuple[
    plt.Figure,
    Tuple[plt.Axes, plt.Axes, plt.Axes],
    Tuple[Optional[plt.AxesImage], Optional[plt.AxesImage]],
    LineCollection,
]:
    """Plot two feature sequences and their transposition-aware alignment.

    The top feature matrix corresponds to the columns of the SSM-TI, while the
    bottom feature matrix corresponds to its rows. Alignment lines connect
    paired frames from ``path`` across the middle axis. Their widths encode
    ``ssm_path`` and their colors encode ``index_ti_path``.

    Parameters
    ----------
    features_x:
        Optional feature matrix corresponding to SSM-TI columns, shaped
        ``(n_features_x, n_frames_x)``. If omitted, the top axis remains blank
        and ``duration_x`` must be provided.
    features_y:
        Optional feature matrix corresponding to SSM-TI rows, shaped
        ``(n_features_y, n_frames_y)``. If omitted, the bottom axis remains
        blank and ``duration_y`` must be provided.
    path:
        Alignment path as ``(row, column)`` index pairs.
        Pairs outside ``duration_x`` or ``duration_y`` are skipped with a
        warning.
    ssm_path:
        Similarity values sampled along ``path``, typically returned by
        :func:`~tovis.structure.get_ssm_index_ti_at_path`.
    index_ti_path:
        Transposition indices sampled along ``path``, typically returned by
        :func:`~tovis.structure.get_ssm_index_ti_at_path`.
    feature_rate_x:
        Feature rate of ``features_x`` in Hz.
    feature_rate_y:
        Feature rate of ``features_y`` in Hz.
    feature_rate_path:
        Frame rate of both path coordinates in Hz. Path positions are converted
        to seconds before being aligned with the feature plots.
    duration_x:
        Duration of sequence X in seconds. Required when ``features_x`` is
        omitted; otherwise inferred from its frame count and feature rate.
        When explicitly provided, this value defines the full horizontal span.
    duration_y:
        Duration of sequence Y in seconds. Required when ``features_y`` is
        omitted; otherwise inferred from its frame count and feature rate.
        When explicitly provided, this value defines the full horizontal span.
    feature_labels_x:
        Optional row labels for ``features_x``.
    feature_labels_y:
        Optional row labels for ``features_y``.
    title_x:
        Title for the top feature axis.
    title_y:
        Title for the bottom feature axis.
    xlabel_x:
        Horizontal-axis label for the top feature axis.
    xlabel_y:
        Horizontal-axis label for the bottom feature axis.
    ylabel_x:
        Vertical-axis label for the top feature axis.
    ylabel_y:
        Vertical-axis label for the bottom feature axis.
    alignment_hop:
        Draw every n-th alignment pair. Must be at least 1.
    ssm_controls_alpha:
        If true, scale line opacity from zero to ``alignment_alpha`` according
        to ``ssm_path``. Otherwise, all lines use ``alignment_alpha``.
    ssm_controls_linewidth:
        If true, scale line widths between ``linewidth_min`` and
        ``linewidth_max`` according to ``ssm_path``. Otherwise, all lines use
        ``linewidth_max``.
    linewidth_min:
        Line width assigned to the lowest similarity.
    linewidth_max:
        Line width assigned to the highest similarity.
    alignment_alpha:
        Opacity of the alignment lines.
    colorbar:
        If true, add a discrete transposition-index colorbar.
    colorbar_label:
        Label for the transposition-index colorbar.
    figsize:
        Figure size.
    height_ratios:
        Relative heights of the top feature, alignment, and bottom feature
        axes.
    kwargs_features_x:
        Optional additional keyword arguments passed to :func:`plot_matrix`
        for the top feature matrix.
    kwargs_features_y:
        Optional additional keyword arguments passed to :func:`plot_matrix`
        for the bottom feature matrix.

    Returns
    -------
    fig, axes, images, alignment_lines:
        Figure; tuple containing the top, middle, and bottom axes; tuple of
        feature-image artists or ``None`` for blank axes; and the
        alignment-line collection.
    """

    if feature_rate_x <= 0 or feature_rate_y <= 0 or feature_rate_path <= 0:
        raise ValueError("feature rates must be positive")
    if path is None or ssm_path is None or index_ti_path is None:
        raise ValueError("path, ssm_path, and index_ti_path must be provided")

    if features_x is not None:
        features_x = _validate_alignment_feature_matrix(features_x, name="features_x")
    if features_y is not None:
        features_y = _validate_alignment_feature_matrix(features_y, name="features_y")
    duration_x = _resolve_alignment_duration(
        features_x,
        feature_rate=feature_rate_x,
        duration=duration_x,
        name="duration_x",
    )
    duration_y = _resolve_alignment_duration(
        features_y,
        feature_rate=feature_rate_y,
        duration=duration_y,
        name="duration_y",
    )

    path, ssm_path, index_ti_path = _validate_alignment_path_data(
        path,
        ssm_path,
        index_ti_path,
        duration_x=duration_x,
        duration_y=duration_y,
        feature_rate_path=feature_rate_path,
    )

    if alignment_hop < 1:
        raise ValueError("alignment_hop must be at least 1")
    if linewidth_min < 0 or linewidth_max < linewidth_min:
        raise ValueError("linewidths must satisfy 0 <= linewidth_min <= linewidth_max")
    if not 0 <= alignment_alpha <= 1:
        raise ValueError("alignment_alpha must be between 0 and 1")
    if len(height_ratios) != 3:
        raise ValueError("height_ratios must contain three values")

    fig, axes_array = plt.subplots(
        3,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.05},
    )
    ax_x, ax_alignment, ax_y = axes_array

    plot_kwargs_x = {
        "feature_rate": feature_rate_x,
        "title": title_x,
        "xlabel": xlabel_x,
        "ylabel": ylabel_x,
        "ylabels": feature_labels_x,
        "colorbar": False,
        "ax": ax_x,
    }
    plot_kwargs_x.update(dict(kwargs_features_x or {}))
    plot_kwargs_x["colorbar"] = False
    plot_kwargs_x["ax"] = ax_x

    plot_kwargs_y = {
        "feature_rate": feature_rate_y,
        "title": title_y,
        "xlabel": xlabel_y,
        "ylabel": ylabel_y,
        "ylabels": feature_labels_y,
        "colorbar": False,
        "ax": ax_y,
    }
    plot_kwargs_y.update(dict(kwargs_features_y or {}))
    plot_kwargs_y["colorbar"] = False
    plot_kwargs_y["ax"] = ax_y

    if features_x is None:
        image_x = None
        _configure_blank_alignment_feature_axis(
            ax_x,
            duration=duration_x,
            title=title_x,
            xlabel=xlabel_x,
            ylabel=ylabel_x,
        )
    else:
        plot_kwargs_x["xlim"] = (0, duration_x)
        _, _, image_x = plot_matrix(features_x, **plot_kwargs_x)

    if features_y is None:
        image_y = None
        _configure_blank_alignment_feature_axis(
            ax_y,
            duration=duration_y,
            title=title_y,
            xlabel="",
            ylabel=ylabel_y,
        )
    else:
        plot_kwargs_y["xlim"] = (0, duration_y)
        _, _, image_y = plot_matrix(features_y, **plot_kwargs_y)

    ax_x.xaxis.set_label_position("top")
    ax_x.xaxis.tick_top()
    ax_x.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax_y.set_xlabel("")

    selected = np.arange(0, path.shape[0], alignment_hop)
    path_selected = path[selected]
    similarity_selected = ssm_path[selected]
    index_selected = index_ti_path[selected]

    if path_selected.shape[0] == 0:
        segments = np.empty((0, 2, 2), dtype=float)
        similarity_weights = np.empty(0, dtype=float)
        linewidths = np.empty(0, dtype=float)
        colors = np.empty((0, 4), dtype=float)
    else:
        segments = np.stack(
            [
                np.column_stack(
                    [
                        ((path_selected[:, 1] + 0.5) / feature_rate_path)
                        / duration_x,
                        np.ones(path_selected.shape[0]),
                    ]
                ),
                np.column_stack(
                    [
                        ((path_selected[:, 0] + 0.5) / feature_rate_path)
                        / duration_y,
                        np.zeros(path_selected.shape[0]),
                    ]
                ),
            ],
            axis=1,
        )
        similarity_weights = _normalize_alignment_similarity(similarity_selected)
        if ssm_controls_linewidth:
            linewidths = linewidth_min + similarity_weights * (linewidth_max - linewidth_min)
        else:
            linewidths = np.full(similarity_weights.shape, linewidth_max)

        colors = mcolors.to_rgba_array(
            np.asarray(get_transposition_index_colors())[index_selected + 1]
        )
        if ssm_controls_alpha:
            colors[:, 3] = similarity_weights * alignment_alpha
        else:
            colors[:, 3] = alignment_alpha

    alignment_lines = LineCollection(
        segments,
        colors=colors,
        linewidths=linewidths,
        transform=ax_alignment.transAxes,
        clip_on=True,
    )
    ax_alignment.add_collection(alignment_lines)
    ax_alignment.set_xlim(0, 1)
    ax_alignment.set_ylim(0, 1)
    ax_alignment.set_axis_off()

    if colorbar:
        colorbar_mappable = plt.cm.ScalarMappable(
            norm=_get_transposition_index_norm(),
            cmap=get_transposition_index_cmap(),
        )
        colorbar_mappable.set_array([])
        colorbar_artist = fig.colorbar(
            colorbar_mappable,
            ax=[ax_x, ax_alignment, ax_y],
            ticks=np.arange(-1, 12),
        )
        colorbar_artist.set_label(colorbar_label)
        colorbar_artist.set_ticklabels([""] + list(TRANSPOSITION_INDEX_LABELS))

    return fig, (ax_x, ax_alignment, ax_y), (image_x, image_y), alignment_lines


def plot_chromagram(
    chromagram: Union[np.ndarray, Sequence[np.ndarray]],
    *,
    direction: str = "horizontal",
    feature_rate: float = 1.0,
    offset: float = 0.0,
    xlim: Optional[Sequence[float]] = None,
    clim: Optional[Sequence[float]] = None,
    title: Optional[str] = "Chromagram",
    xlabel: str = "Time (seconds)",
    ylabel: str = "Pitch class",
    chroma_labels: Sequence[str] = CHROMA_LABELS,
    show_natural_pitch_labels: bool = False,
    ytick_label_hop: int = 1,
    grid_lines: bool = True,
    overlay_labels: Optional[Sequence[str]] = None,
    colorbar: bool = True,
    colorbar_label: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 4),
    cmap: str = "gray_r",
    interpolation: str = "None",
) -> Tuple[plt.Figure, plt.Axes, plt.AxesImage]:
    """Plot a 12-bin chromagram.

    If ``chromagram`` is a list of two or three matrices, a CMY overlay is
    drawn via ``plot_cmy_overlay``. All matrices must have one row per
    ``chroma_labels`` entry. With ``direction="vertical"``, time runs along
    the y-axis and chroma labels are shown on the x-axis.

    Parameters
    ----------
    chromagram:
        Chromagram matrix of shape ``(12, n_frames)``, or a list of two or
        three such matrices.
    direction:
        Time-axis direction, either ``"horizontal"`` or ``"vertical"``.
    feature_rate:
        Number of chromagram frames per second.
    offset:
        Time value of the first chromagram frame in seconds.
    xlim:
        Optional visible time range in seconds.
    clim:
        Optional color limits for single-matrix plots.
    title:
        Axis title.
    xlabel:
        Horizontal-axis label.
    ylabel:
        Vertical-axis label.
    chroma_labels:
        Labels for the 12 chroma rows.
    show_natural_pitch_labels:
        If true, label only the natural-pitch chroma rows:
        C, D, E, F, G, A, and B.
    ytick_label_hop:
        Show only every n-th y-axis label.
    grid_lines:
        If true, overlay light reference lines between feature rows using
        :func:`overlay_lines`. Lines are horizontal for
        ``direction="horizontal"`` and vertical for ``direction="vertical"``.
    overlay_labels:
        Optional legend labels for CMY overlays.
    colorbar:
        If true, add a colorbar for single-matrix plots.
    colorbar_label:
        Label for the colorbar.
    ax:
        Existing axis to draw into. If omitted, a new figure and axis are
        created.
    figsize:
        Figure size used when creating a new figure.
    cmap:
        Colormap passed to ``plot_matrix``.
    interpolation:
        Image interpolation passed to ``plot_matrix``.

    Returns
    -------
    fig, ax, image:
        The Matplotlib figure, axis, and image artist.
    """

    if _is_feature_matrix_sequence(chromagram):
        chromagrams = [np.asarray(feature) for feature in chromagram]
        if any(feature.shape[0] != len(chroma_labels) for feature in chromagrams):
            raise ValueError("all chromagrams must match the number of chroma labels")

        fig, ax, image = plot_cmy_overlay(
            chromagrams,
            labels=overlay_labels,
            direction=direction,
            feature_rate=feature_rate,
            offset=offset,
            frequency_rate=1.0,
            xlim=xlim,
            ylim=(-0.5, len(chroma_labels) - 0.5),
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            ylabels=chroma_labels,
            ytick_label_hop=ytick_label_hop,
            ax=ax,
            figsize=figsize,
            interpolation=interpolation,
        )
        if show_natural_pitch_labels:
            _set_natural_pitch_tick_labels(
                ax,
                chroma_labels,
                hop=ytick_label_hop,
                direction=direction,
            )
        if grid_lines:
            overlay_lines(ax, direction=direction)
        return fig, ax, image

    chromagram = np.asarray(chromagram)
    if chromagram.shape[0] != len(chroma_labels):
        raise ValueError("chromagram first axis must match the number of chroma labels")

    fig, ax, image = plot_matrix(
        chromagram,
        direction=direction,
        feature_rate=feature_rate,
        offset=offset,
        frequency_rate=1.0,
        xlim=xlim,
        ylim=(-0.5, len(chroma_labels) - 0.5),
        clim=clim,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ylabels=chroma_labels,
        ytick_label_hop=ytick_label_hop,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
        ax=ax,
        figsize=figsize,
        cmap=cmap,
        interpolation=interpolation,
    )
    if show_natural_pitch_labels:
        _set_natural_pitch_tick_labels(
            ax,
            chroma_labels,
            hop=ytick_label_hop,
            direction=direction,
        )
    if grid_lines:
        overlay_lines(ax, direction=direction)
    return fig, ax, image


def plot_scale_likelihood(
    scale_likelihood: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    *,
    major_likelihood: Optional[np.ndarray] = None,
    minor_likelihood: Optional[np.ndarray] = None,
    mode_contrast: Optional[np.ndarray] = None,
    direction: str = "horizontal",
    feature_rate: float = 1.0,
    offset: float = 0.0,
    xlim: Optional[Sequence[float]] = None,
    clim: Optional[Sequence[float]] = None,
    title: Optional[str] = "Scale likelihood",
    xlabel: str = "Time (seconds)",
    ylabel: str = "Scale",
    scale_label_version: str = "absolute_diatonic",
    scale_labels: Optional[Sequence[str]] = None,
    sort_by_fifths: bool = False,
    global_key: Optional[str] = None,
    ytick_label_hop: int = 1,
    grid_lines: bool = True,
    overlay_labels: Optional[Sequence[str]] = None,
    colorbar: bool = True,
    colorbar_label: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 4),
    cmap: str = "gray_r",
    mode_contrast_cmap: Optional[mcolors.Colormap] = None,
    mode_contrast_clim: Sequence[float] = (-1.0, 1.0),
    mode_contrast_colorbar_label: str = "Major-minor contrast",
    interpolation: str = "None",
) -> Tuple[plt.Figure, plt.Axes, plt.AxesImage]:
    """Plot a 12-bin scale-likelihood matrix.

    The rows are labelled with scale labels from ``get_scale_labels`` unless
    explicit labels are provided. Lists of two or three matrices are shown as a
    CMY overlay. Optionally, rows can be reordered by circle-of-fifths
    distance. With ``direction="vertical"``, time runs along the y-axis and
    scale labels are shown on the x-axis.

    Parameters
    ----------
    scale_likelihood:
        Scale-likelihood matrix of shape ``(12, n_frames)``, or a list of two
        or three matrices to plot as a CMY overlay. In major/minor contrast
        mode, this matrix is used as the image alpha channel, so the ordinary
        diatonic likelihood remains visible as intensity.
    major_likelihood:
        Optional major-scale likelihood matrix. If provided together with
        ``minor_likelihood``, the plotted hue is determined by
        ``major_likelihood - minor_likelihood``.
    minor_likelihood:
        Optional minor-scale likelihood matrix. Must have the same shape as
        ``major_likelihood``.
    mode_contrast:
        Optional precomputed major/minor contrast matrix for the plot hue.
        Values near ``-1`` are plotted red, values near ``0`` black, and
        values near ``1`` blue.
    direction:
        Time-axis direction, either ``"horizontal"`` or ``"vertical"``.
    feature_rate:
        Number of scale-likelihood frames per second.
    offset:
        Time value of the first scale-likelihood frame in seconds.
    xlim:
        Optional visible time range in seconds.
    clim:
        Optional color limits for single-matrix plots.
    title:
        Axis title.
    xlabel:
        Horizontal-axis label.
    ylabel:
        Vertical-axis label.
    scale_label_version:
        Label version passed to ``get_scale_labels`` when ``scale_labels`` is
        not provided. Options are ``"absolute_diatonic"`` and
        ``"relative_diatonic"``.
    scale_labels:
        Optional explicit labels for the 12 scale rows.
    sort_by_fifths:
        If true, reorder rows and labels around the circle of fifths.
    global_key:
        Optional global key. For ``scale_label_version="relative_diatonic"``,
        this key is labelled as ``"0"``. When ``sort_by_fifths`` is true, it
        is also placed near the center of the vertical axis.
    ytick_label_hop:
        Show only every n-th y-axis label. If ``global_key`` is used, the
        centered label is always shown.
    grid_lines:
        If true, overlay light reference lines between feature rows using
        :func:`overlay_lines`. Lines are horizontal for
        ``direction="horizontal"`` and vertical for ``direction="vertical"``.
    overlay_labels:
        Optional legend labels for CMY overlays.
    colorbar:
        If true, add a colorbar for single-matrix plots.
    colorbar_label:
        Label for the colorbar.
    ax:
        Existing axis to draw into. If omitted, a new figure and axis are
        created.
    figsize:
        Figure size used when creating a new figure.
    cmap:
        Colormap passed to ``plot_matrix``.
    mode_contrast_cmap:
        Colormap used when plotting major/minor contrast. If omitted, a
        red-black-blue colormap is used.
    mode_contrast_clim:
        Color limits for major/minor contrast plots. The default ``(-1, 1)``
        maps minor evidence to red and major evidence to blue.
    mode_contrast_colorbar_label:
        Colorbar label used for major/minor contrast plots when
        ``colorbar_label`` is empty.
    interpolation:
        Image interpolation passed to ``plot_matrix``.

    Returns
    -------
    fig, ax, image:
        The Matplotlib figure, axis, and image artist.
    """

    if scale_labels is None:
        scale_labels = get_scale_labels(
            scale_label_version,
            global_key="C" if global_key is None else global_key,
        )
    scale_labels = list(scale_labels)
    mode_contrast = _resolve_mode_contrast(
        mode_contrast,
        major_likelihood=major_likelihood,
        minor_likelihood=minor_likelihood,
    )
    use_mode_contrast = mode_contrast is not None

    if scale_likelihood is None and not use_mode_contrast:
        raise ValueError(
            "scale_likelihood, mode_contrast, or both major_likelihood and "
            "minor_likelihood must be provided"
        )
    if use_mode_contrast and _is_feature_matrix_sequence(scale_likelihood):
        raise ValueError("CMY overlays cannot be combined with mode_contrast")

    if not use_mode_contrast and _is_feature_matrix_sequence(scale_likelihood):
        scale_likelihoods = [np.asarray(feature) for feature in scale_likelihood]
        if any(feature.shape[0] != len(scale_labels) for feature in scale_likelihoods):
            raise ValueError("all scale-likelihood matrices must match the number of scale labels")

        if sort_by_fifths:
            scale_likelihoods, scale_labels = _sort_scale_features_and_labels_by_fifths(
                scale_likelihoods,
                scale_labels,
                global_key=global_key,
            )

        force_ytick_labels = [scale_labels[len(scale_labels) // 2]] if sort_by_fifths and global_key is not None else None
        fig, ax, image = plot_cmy_overlay(
            scale_likelihoods,
            labels=overlay_labels,
            direction=direction,
            feature_rate=feature_rate,
            offset=offset,
            frequency_rate=1.0,
            xlim=xlim,
            ylim=(-0.5, len(scale_labels) - 0.5),
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            ylabels=scale_labels,
            ytick_label_hop=ytick_label_hop,
            force_ytick_labels=force_ytick_labels,
            ax=ax,
            figsize=figsize,
            interpolation=interpolation,
        )
        if grid_lines:
            overlay_lines(ax, direction=direction)
        return fig, ax, image

    alpha_likelihood = None
    if use_mode_contrast:
        alpha_likelihood = _resolve_mode_contrast_alpha(
            scale_likelihood,
            major_likelihood=major_likelihood,
            minor_likelihood=minor_likelihood,
            mode_contrast=mode_contrast,
        )
        scale_likelihood = np.asarray(mode_contrast, dtype=float)
    else:
        scale_likelihood = np.asarray(scale_likelihood, dtype=float)
    if scale_likelihood.ndim != 2:
        raise ValueError("scale_likelihood must be a two-dimensional matrix")
    if scale_likelihood.shape[0] != len(scale_labels):
        raise ValueError("scale_likelihood first axis must match the number of scale labels")
    if alpha_likelihood is not None and alpha_likelihood.shape != scale_likelihood.shape:
        raise ValueError("scale_likelihood must have the same shape as mode_contrast")

    if sort_by_fifths:
        features_to_sort = [scale_likelihood]
        if alpha_likelihood is not None:
            features_to_sort.append(alpha_likelihood)
        sorted_features, scale_labels = _sort_scale_features_and_labels_by_fifths(
            features_to_sort,
            scale_labels,
            global_key=global_key,
        )
        scale_likelihood = sorted_features[0]
        if alpha_likelihood is not None:
            alpha_likelihood = sorted_features[1]

    force_ytick_labels = [scale_labels[len(scale_labels) // 2]] if sort_by_fifths and global_key is not None else None
    if use_mode_contrast:
        plot_cmap = mode_contrast_cmap or get_major_minor_contrast_cmap()
        plot_clim = tuple(clim) if clim is not None else tuple(mode_contrast_clim)
        plot_colorbar_label = colorbar_label or mode_contrast_colorbar_label
        scale_likelihood = _mode_contrast_to_rgba(
            scale_likelihood,
            alpha_likelihood,
            cmap=plot_cmap,
            clim=plot_clim,
        )
        matrix_colorbar = False
    else:
        plot_cmap = cmap
        plot_clim = clim
        plot_colorbar_label = colorbar_label
        matrix_colorbar = colorbar
    fig, ax, image = plot_matrix(
        scale_likelihood,
        direction=direction,
        feature_rate=feature_rate,
        offset=offset,
        frequency_rate=1.0,
        xlim=xlim,
        ylim=(-0.5, len(scale_labels) - 0.5),
        clim=None if use_mode_contrast else plot_clim,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ylabels=scale_labels,
        ytick_label_hop=ytick_label_hop,
        force_ytick_labels=force_ytick_labels,
        colorbar=matrix_colorbar,
        colorbar_label=plot_colorbar_label,
        ax=ax,
        figsize=figsize,
        cmap=plot_cmap,
        interpolation=interpolation,
    )
    if use_mode_contrast and colorbar:
        colorbar_mappable = plt.cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=plot_clim[0], vmax=plot_clim[1]),
            cmap=plot_cmap,
        )
        colorbar_mappable.set_array([])
        colorbar_artist = fig.colorbar(colorbar_mappable, ax=ax)
        colorbar_artist.set_label(plot_colorbar_label)
    if grid_lines:
        overlay_lines(ax, direction=direction)
    return fig, ax, image


def plot_tempo_curve(
    measure_annotation: pd.DataFrame,
    *,
    time_column: str = "time",
    measure_column: str = "measure",
    x_axis: str = "time",
    ax: Optional[plt.Axes] = None,
    use_twin_axis: bool = False,
    figsize: Tuple[float, float] = (10, 3),
    title: Optional[str] = "Tempo curve",
    xlabel: Optional[str] = None,
    ylabel: str = "Measures per second",
    smoothing_length_measures: Optional[float] = None,
    smoothing_method: str = "mean",
    color: str = "tab:blue",
    linewidth: float = 1.5,
    grid: bool = True,
) -> Tuple[plt.Figure, plt.Axes, plt.Line2D]:
    """Plot a local tempo curve from time/measure annotations.

    Tempo is computed as the finite-difference slope ``d(measure) / d(time)``
    between adjacent annotation points and plotted at interval midpoints. Use
    ``x_axis="time"`` for seconds or ``x_axis="measure"`` for measure positions.
    If ``use_twin_axis`` is true, tempo is plotted on a right-side y-axis and
    that tempo axis is returned.

    Parameters
    ----------
    measure_annotation:
        Annotation table containing physical time and measure-number columns.
    time_column:
        Column name for seconds; passed to
        :func:`~tovis.utils.get_tempo_curve`.
    measure_column:
        Column name for measure positions; passed to
        :func:`~tovis.utils.get_tempo_curve`.
    x_axis:
        Horizontal axis for the plot, either ``"time"`` or ``"measure"``.
    ax:
        Existing axis to draw into. If omitted, a new figure and axis are
        created.
    use_twin_axis:
        If true, draw the tempo curve on a right-side y-axis created from
        ``ax``.
    figsize:
        Figure size used when creating a new figure.
    title:
        Plot title.
    xlabel:
        Horizontal axis label.
    ylabel:
        Vertical axis label for tempo.
    smoothing_length_measures:
        Optional smoothing-window length in measures passed to
        :func:`~tovis.utils.get_tempo_curve`.
    smoothing_method:
        Smoothing aggregation method passed to
        :func:`~tovis.utils.get_tempo_curve`.
    color:
        Line color.
    linewidth:
        Line width.
    grid:
        If true, draw a light grid on the base axis.

    Returns
    -------
    fig, tempo_ax, line:
        The Matplotlib figure, the axis containing the tempo curve, and the
        tempo line artist.
    """

    axis_seconds, axis_measures, tempo = get_tempo_curve(
        measure_annotation,
        time_column=time_column,
        measure_column=measure_column,
        smoothing_length_measures=smoothing_length_measures,
        smoothing_method=smoothing_method,
    )

    if x_axis == "time":
        x_values = axis_seconds
        xlabel = "Time (seconds)" if xlabel is None else xlabel
    elif x_axis == "measure":
        x_values = axis_measures
        xlabel = "Measure" if xlabel is None else xlabel
    else:
        raise ValueError('x_axis must be "time" or "measure"')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    tempo_ax = ax.twinx() if use_twin_axis else ax

    (line,) = tempo_ax.plot(x_values, tempo, color=color, linewidth=linewidth)
    if title is not None:
        ax.set_title(title)
    if not use_twin_axis:
        ax.set_xlabel(xlabel)
    tempo_ax.set_ylabel(ylabel)
    if use_twin_axis:
        tempo_ax.tick_params(axis="y", colors=color)
        tempo_ax.yaxis.label.set_color(color)
    if grid:
        ax.grid(True, alpha=0.3)

    return fig, tempo_ax, line


def _get_annotation_values(
    annotation: pd.DataFrame,
    column_name: str,
    fallback_position: int,
) -> Sequence[Any]:
    """Return an annotation column, falling back to positional access.

    Parameters
    ----------
    annotation:
        Annotation table.
    column_name:
        Preferred column name.
    fallback_position:
        Positional column index used when ``column_name`` is absent.

    Returns
    -------
    sequence
        Values from the selected annotation column.
    """

    if column_name in annotation.columns:
        return annotation[column_name].to_numpy()

    return annotation.iloc[:, fallback_position].to_numpy()


def _get_correspondence_part_names(row: pd.Series) -> list[str]:
    """Return nonempty part names from one correspondence row."""

    part_names = []
    for value in row.to_numpy():
        if pd.isna(value):
            continue
        name = _normalize_correspondence_part_name(value)
        if name:
            part_names.append(name)

    return part_names


def _normalize_correspondence_part_name(value: Any) -> str:
    """Normalize a part name read from a correspondence annotation table."""

    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _get_correspondence_color(
    harmony_x: Any,
    harmony_y: Any,
    *,
    colors: Sequence[str],
    modulation_label: str,
) -> str:
    """Return the transposition color for a pair of harmony labels."""

    harmony_x = str(harmony_x).strip()
    harmony_y = str(harmony_y).strip()
    if harmony_x == modulation_label or harmony_y == modulation_label:
        return "k"

    try:
        pitch_x = key_label_to_pitch_class(harmony_x)
        pitch_y = key_label_to_pitch_class(harmony_y)
    except ValueError:
        return "k"

    transposition = (pitch_y - pitch_x) % 12
    return colors[int(transposition)]


def _get_hopped_tick_labels(
    labels: Sequence[str],
    *,
    hop: int,
    force_labels: Optional[Sequence[str]],
) -> Tuple[np.ndarray, list[str]]:
    """Select y-axis labels according to a hop while preserving forced labels.

    Parameters
    ----------
    labels:
        Full label sequence.
    hop:
        Label stride. A value of 1 keeps all labels.
    force_labels:
        Optional labels that must be retained even if the hop would skip them.

    Returns
    -------
    tick_positions, tick_labels:
        Integer row positions and corresponding labels.
    """

    if hop < 1:
        raise ValueError("ytick_label_hop must be at least 1")

    label_indices = set(range(0, len(labels), hop))
    if force_labels is not None:
        normalized_force_labels = {_normalize_label(label) for label in force_labels}
        for index, label in enumerate(labels):
            if _normalize_label(label) in normalized_force_labels:
                label_indices.add(index)

    tick_positions = np.array(sorted(label_indices))
    tick_labels = [labels[index] for index in tick_positions]
    return tick_positions, tick_labels


def _set_natural_pitch_tick_labels(
    ax: plt.Axes,
    labels: Sequence[str],
    *,
    hop: int,
    direction: str,
) -> None:
    """Set feature-axis ticks to chroma rows corresponding to natural pitches."""

    if len(labels) != 12:
        raise ValueError("show_natural_pitch_labels requires 12 chroma labels")
    if hop < 1:
        raise ValueError("ytick_label_hop must be at least 1")
    if direction not in {"horizontal", "vertical"}:
        raise ValueError('direction must be "horizontal" or "vertical"')

    natural_pitch_indices = np.array([0, 2, 4, 5, 7, 9, 11])
    selected_indices = natural_pitch_indices[np.arange(natural_pitch_indices.size) % hop == 0]
    selected_labels = [labels[index] for index in selected_indices]
    if direction == "horizontal":
        ax.set_yticks(selected_indices)
        ax.set_yticklabels(selected_labels)
    else:
        ax.set_xticks(selected_indices)
        ax.set_xticklabels(selected_labels)


def _filter_positions_by_limits(
    positions: Sequence[float],
    labels: Sequence[str],
    limits: Tuple[float, float],
) -> Tuple[list[float], list[str]]:
    """Keep annotation positions and labels that fall inside axis limits."""

    lower, upper = sorted(limits)
    filtered_positions = []
    filtered_labels = []

    for position, label in zip(positions, labels):
        if lower <= position <= upper:
            filtered_positions.append(position)
            filtered_labels.append(label)

    return filtered_positions, filtered_labels


def _annotation_label_axis_position(label_location: str) -> float:
    """Map a label-location name to an axes-coordinate y/x position."""

    if label_location == "bottom":
        return 0.02
    if label_location == "top":
        return 0.98

    raise ValueError('label_location must be "bottom" or "top"')


def _annotation_label_alignment(label_location: str, direction: str) -> dict[str, str]:
    """Return text alignment that keeps annotation labels inside the axes."""

    _annotation_label_axis_position(label_location)

    if direction == "vertical":
        return {
            "horizontalalignment": "left",
            "verticalalignment": label_location,
        }
    if direction == "horizontal":
        return {
            "horizontalalignment": "left" if label_location == "bottom" else "right",
            "verticalalignment": "bottom",
        }

    raise ValueError('direction must be "vertical" or "horizontal"')


def _is_feature_matrix_sequence(value: Any) -> bool:
    """Return true if ``value`` should be treated as multiple matrices."""

    return isinstance(value, (list, tuple))


def _get_scale_row_lookup(
    scale_labels: Sequence[str],
    *,
    sort_by_fifths: bool,
    global_key: Optional[str],
) -> dict[int, int]:
    """Map chromatic diatonic-center pitch classes to displayed row positions."""

    if len(scale_labels) != 12:
        raise ValueError("scale labels must contain exactly 12 entries")

    if not sort_by_fifths:
        return {pitch_class: pitch_class for pitch_class in range(12)}

    fifth_indices, shift = _get_fifths_order(scale_labels, global_key=global_key)
    sorted_pitch_classes = list(fifth_indices)
    if shift:
        sorted_pitch_classes = _roll_sequence(sorted_pitch_classes, shift)

    return {
        int(pitch_class): row_position
        for row_position, pitch_class in enumerate(sorted_pitch_classes)
    }


def _get_localkey_mode(localkey: str) -> str:
    """Infer ``"major"`` or ``"minor"`` from a local-key label."""

    label = localkey.strip()
    label_lower = label.lower()
    if ":" in label_lower:
        mode = label_lower.split(":", maxsplit=1)[1]
        if mode in {"min", "minor", "m"}:
            return "minor"
        return "major"
    if label.endswith("m") and len(label) > 1:
        return "minor"
    return "major"


def _resolve_mode_contrast(
    mode_contrast: Optional[np.ndarray],
    *,
    major_likelihood: Optional[np.ndarray],
    minor_likelihood: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Return a major/minor contrast matrix or ``None`` if not requested."""

    if mode_contrast is not None and (
        major_likelihood is not None or minor_likelihood is not None
    ):
        raise ValueError(
            "provide either mode_contrast or major_likelihood/minor_likelihood, not both"
        )
    if mode_contrast is not None:
        contrast = np.asarray(mode_contrast, dtype=float)
    elif major_likelihood is not None or minor_likelihood is not None:
        if major_likelihood is None or minor_likelihood is None:
            raise ValueError(
                "major_likelihood and minor_likelihood must be provided together"
            )
        major_likelihood = np.asarray(major_likelihood, dtype=float)
        minor_likelihood = np.asarray(minor_likelihood, dtype=float)
        if major_likelihood.shape != minor_likelihood.shape:
            raise ValueError("major_likelihood and minor_likelihood must have the same shape")
        contrast = major_likelihood - minor_likelihood
    else:
        return None

    if contrast.ndim != 2:
        raise ValueError("mode_contrast must be a two-dimensional matrix")
    return contrast


def _resolve_mode_contrast_alpha(
    scale_likelihood: Optional[np.ndarray],
    *,
    major_likelihood: Optional[np.ndarray],
    minor_likelihood: Optional[np.ndarray],
    mode_contrast: np.ndarray,
) -> np.ndarray:
    """Return the alpha matrix for a major/minor contrast plot."""

    if scale_likelihood is not None:
        alpha = np.asarray(scale_likelihood, dtype=float)
    elif major_likelihood is not None and minor_likelihood is not None:
        alpha = np.maximum(
            np.asarray(major_likelihood, dtype=float),
            np.asarray(minor_likelihood, dtype=float),
        )
    else:
        alpha = np.ones_like(mode_contrast, dtype=float)

    if alpha.ndim != 2:
        raise ValueError("scale_likelihood must be a two-dimensional matrix")
    return alpha


def _mode_contrast_to_rgba(
    mode_contrast: np.ndarray,
    alpha_likelihood: np.ndarray,
    *,
    cmap: mcolors.Colormap,
    clim: Sequence[float],
) -> np.ndarray:
    """Encode major/minor contrast as hue and diatonic likelihood as alpha."""

    if len(clim) != 2:
        raise ValueError("mode_contrast_clim must contain two values")
    vmin, vmax = float(clim[0]), float(clim[1])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("mode_contrast_clim must be a finite increasing pair")

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cmap(norm(mode_contrast))
    rgba[:, :, 3] = np.clip(alpha_likelihood, 0.0, 1.0)
    return rgba


def _format_measure_label(measure: float) -> str:
    """Format an integer-like measure number without a decimal part."""

    if measure.is_integer():
        return str(int(measure))

    return f"{measure:g}"


def _get_cmy_legend_handles(labels: Sequence[str]) -> list[mpatches.Patch]:
    """Create legend patches for two- or three-channel CMY overlays."""

    if len(labels) == 2:
        colors = [
            [0, 1, 1],
            [1, 0, 1],
            [0, 0, 1],
        ]
        legend_labels = [
            labels[0],
            labels[1],
            f"{labels[0]} + {labels[1]}",
        ]
    elif len(labels) == 3:
        colors = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]
        legend_labels = [
            labels[0],
            labels[1],
            labels[2],
            f"{labels[0]} + {labels[1]}",
            f"{labels[0]} + {labels[2]}",
            f"{labels[1]} + {labels[2]}",
            "all",
        ]
    else:
        raise ValueError("labels must contain two or three entries")

    return [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(colors, legend_labels)
    ]


def _get_transposition_index_norm() -> mcolors.BoundaryNorm:
    """Return a discrete normalization for transposition indices -1 to 11."""

    return mcolors.BoundaryNorm(np.arange(-1.5, 12.5, 1), ncolors=13)


def _validate_transposition_index_matrix(index_ti: np.ndarray) -> np.ndarray:
    """Return a valid integer transposition-index matrix."""

    index_ti = np.asarray(index_ti)
    if index_ti.ndim != 2:
        raise ValueError("index_ti must be a two-dimensional matrix")
    if np.any(index_ti < -1) or np.any(index_ti > 11):
        raise ValueError("index_ti values must be between -1 and 11")
    return index_ti.astype(int)


def _sort_transposition_indices_by_fifths(
    index_ti_path: np.ndarray,
) -> Tuple[np.ndarray, list[str]]:
    """Map transposition indices to fifths-sorted positions centered on zero."""

    if not np.all(np.equal(index_ti_path, np.floor(index_ti_path))):
        raise ValueError("index_ti_path must contain integer values")
    if np.any(index_ti_path < 0) or np.any(index_ti_path > 11):
        raise ValueError("index_ti_path values must be between 0 and 11 after filtering")

    fifths_order = np.mod(np.arange(12) * 7, 12)
    fifths_order = np.roll(fifths_order, len(fifths_order) // 2)
    display_positions = np.empty(12, dtype=int)
    display_positions[fifths_order] = np.arange(12)
    y_values = display_positions[index_ti_path.astype(int)]
    ytick_labels = [
        str(distance) if distance <= 0 else f"+{distance}"
        for distance in range(-6, 6)
    ]
    return y_values, ytick_labels


def _validate_alignment_feature_matrix(features: np.ndarray, *, name: str) -> np.ndarray:
    """Return a nonempty two-dimensional feature matrix."""

    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if min(features.shape) == 0:
        raise ValueError(f"{name} must not be empty")
    return features


def _plot_aligned_feature_x(
    features: Optional[np.ndarray],
    *,
    ax: plt.Axes,
    duration: float,
    feature_rate: float,
    offset: float,
    labels: Optional[Sequence[str]],
    xlabel: str,
    ylabel: str,
    kwargs: Optional[Mapping[str, Any]],
) -> Optional[plt.AxesImage]:
    """Plot or configure a feature panel aligned to an SSM x-axis."""

    if features is None:
        ax.set_xlim(offset, offset + duration)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return None

    plot_kwargs = {
        "feature_rate": feature_rate,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "ylabels": labels,
        "colorbar": False,
        "ax": ax,
    }
    plot_kwargs.update(dict(kwargs or {}))
    plot_kwargs["colorbar"] = False
    plot_kwargs["ax"] = ax
    _, _, image = plot_matrix(features, **plot_kwargs)
    extent = image.get_extent()
    feature_duration = features.shape[1] / feature_rate
    image.set_extent([offset, offset + feature_duration, extent[2], extent[3]])
    return image


def _plot_aligned_feature_y(
    features: Optional[np.ndarray],
    *,
    ax: plt.Axes,
    duration: float,
    feature_rate: float,
    offset: float,
    labels: Optional[Sequence[str]],
    xlabel: str,
    ylabel: str,
    kwargs: Optional[Mapping[str, Any]],
) -> Optional[plt.AxesImage]:
    """Plot or configure a rotated feature panel aligned to an SSM y-axis."""

    if features is None:
        ax.set_ylim(offset, offset + duration)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return None

    plot_kwargs = {
        "feature_rate": 1.0,
        "frequency_rate": feature_rate,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "xlabels": labels,
        "colorbar": False,
        "ax": ax,
    }
    plot_kwargs.update(dict(kwargs or {}))
    plot_kwargs["colorbar"] = False
    plot_kwargs["ax"] = ax
    _, _, image = plot_matrix(features.T, **plot_kwargs)
    extent = image.get_extent()
    feature_duration = features.shape[1] / feature_rate
    image.set_extent([extent[0], extent[1], offset, offset + feature_duration])
    return image


def _align_feature_axes_to_ssm(
    ax_ssm: plt.Axes,
    *,
    ax_features_x: plt.Axes,
    ax_features_y: plt.Axes,
    ax_corner: plt.Axes,
) -> None:
    """Align surrounding panels to the SSM with equal physical panel gaps."""

    ssm_position = ax_ssm.get_position()
    x_position = ax_features_x.get_position()
    y_position = ax_features_y.get_position()
    vertical_gap = max(ssm_position.y0 - x_position.y1, 0.0)
    horizontal_gap = vertical_gap * ax_ssm.figure.get_figheight() / ax_ssm.figure.get_figwidth()
    y_left = ssm_position.x0 - horizontal_gap - y_position.width

    ax_features_x.set_position(
        [ssm_position.x0, x_position.y0, ssm_position.width, x_position.height]
    )
    ax_features_y.set_position(
        [y_left, ssm_position.y0, y_position.width, ssm_position.height]
    )
    ax_corner.set_position(
        [y_left, x_position.y0, y_position.width, x_position.height]
    )


def _resolve_alignment_duration(
    features: Optional[np.ndarray],
    *,
    feature_rate: float,
    duration: Optional[float],
    name: str,
) -> float:
    """Return an explicit duration or infer it from a feature matrix."""

    if duration is None:
        if features is None:
            raise ValueError(f"{name} must be provided when its feature matrix is omitted")
        duration = features.shape[1] / feature_rate

    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return duration


def _configure_blank_alignment_feature_axis(
    ax: plt.Axes,
    *,
    duration: float,
    title: Optional[str],
    xlabel: str,
    ylabel: str,
) -> None:
    """Configure a blank feature axis spanning a known sequence duration."""

    ax.set_xlim(0, duration)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)


def _validate_alignment_path_data(
    path: np.ndarray,
    ssm_path: np.ndarray,
    index_ti_path: np.ndarray,
    *,
    duration_x: float,
    duration_y: float,
    feature_rate_path: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate path-aligned similarity and transposition-index values."""

    path = np.asarray(path, dtype=int)
    ssm_path = np.asarray(ssm_path, dtype=float)
    index_ti_path = np.asarray(index_ti_path)

    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must have shape (n_steps, 2)")
    if path.shape[0] == 0:
        raise ValueError("path must contain at least one alignment pair")
    if ssm_path.ndim != 1 or ssm_path.shape[0] != path.shape[0]:
        raise ValueError("ssm_path must be one-dimensional and match path length")
    if index_ti_path.ndim != 1 or index_ti_path.shape[0] != path.shape[0]:
        raise ValueError("index_ti_path must be one-dimensional and match path length")
    if not np.all(np.isfinite(ssm_path)):
        raise ValueError("ssm_path must contain only finite values")
    if not np.all(np.equal(index_ti_path, np.floor(index_ti_path))):
        raise ValueError("index_ti_path must contain integer values")
    if np.any(index_ti_path < -1) or np.any(index_ti_path > 11):
        raise ValueError("index_ti_path values must be between -1 and 11")
    if np.any(path < 0):
        raise ValueError("path indices must be non-negative")
    row_positions = (path[:, 0] + 0.5) / feature_rate_path
    column_positions = (path[:, 1] + 0.5) / feature_rate_path
    rows_within_duration = row_positions <= duration_y
    columns_within_duration = column_positions <= duration_x
    within_duration = rows_within_duration & columns_within_duration

    if not np.all(rows_within_duration):
        n_skipped = int(np.count_nonzero(~rows_within_duration))
        warnings.warn(
            f"Skipping {n_skipped} alignment pair(s): "
            "path row positions exceed the duration of features_y",
            RuntimeWarning,
            stacklevel=2,
        )
    if not np.all(columns_within_duration):
        n_skipped = int(np.count_nonzero(~columns_within_duration))
        warnings.warn(
            f"Skipping {n_skipped} alignment pair(s): "
            "path column positions exceed the duration of features_x",
            RuntimeWarning,
            stacklevel=2,
        )

    path = path[within_duration]
    ssm_path = ssm_path[within_duration]
    index_ti_path = index_ti_path[within_duration]

    return path, ssm_path, index_ti_path.astype(int)


def _normalize_alignment_similarity(similarity: np.ndarray) -> np.ndarray:
    """Normalize alignment similarities to the interval [0, 1]."""

    value_min = np.min(similarity)
    value_range = np.max(similarity) - value_min
    if value_range == 0:
        return np.ones_like(similarity, dtype=float)
    return (similarity - value_min) / value_range


def _similarity_to_alpha(ssm_ti: np.ndarray, *, alpha_min: float) -> np.ndarray:
    """Convert similarity values to continuous alpha values."""

    if not 0 <= alpha_min <= 1:
        raise ValueError("alpha_min must be between 0 and 1")

    similarity = np.asarray(ssm_ti, dtype=float)
    alpha = np.zeros_like(similarity, dtype=float)
    finite_nonnegative = np.isfinite(similarity) & (similarity >= 0)
    if not np.any(finite_nonnegative):
        return alpha

    values = similarity[finite_nonnegative]
    value_min = np.min(values)
    value_range = np.max(values) - value_min
    if value_range == 0:
        alpha[finite_nonnegative] = 1.0
    else:
        alpha[finite_nonnegative] = (values - value_min) / value_range
        alpha[finite_nonnegative] = alpha_min + (1.0 - alpha_min) * alpha[finite_nonnegative]

    return np.clip(alpha, 0.0, 1.0)


def _normalize_marker_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize marker weights to the interval [0, 1]."""

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("ssm_path must be one-dimensional")
    if weights.size == 0:
        return weights

    value_range = np.max(weights) - np.min(weights)
    if value_range == 0:
        return np.ones_like(weights)

    return (weights - np.min(weights)) / value_range


def _pad_feature_matrices_to_longest(features: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Right-pad feature matrices with zeros to a common frame length."""

    max_length = max(feature.shape[1] for feature in features)
    padded_features = []

    for feature in features:
        pad_width = max_length - feature.shape[1]
        if pad_width == 0:
            padded_features.append(feature)
        else:
            padded_features.append(
                np.pad(
                    feature,
                    pad_width=((0, 0), (0, pad_width)),
                    mode="constant",
                )
            )

    return padded_features


def _sort_scale_features_and_labels_by_fifths(
    scale_features: Union[np.ndarray, Sequence[np.ndarray]],
    scale_labels: Sequence[str],
    *,
    global_key: Optional[str],
) -> Tuple[Union[np.ndarray, list[np.ndarray]], list[str]]:
    """Sort one or more scale matrices and labels by circle-of-fifths order."""

    fifth_indices, shift = _get_fifths_order(scale_labels, global_key=global_key)
    scale_features_is_sequence = _is_feature_matrix_sequence(scale_features)
    feature_list = list(scale_features) if scale_features_is_sequence else [scale_features]
    sorted_features = [
        np.take(feature, fifth_indices, axis=0)
        for feature in feature_list
    ]
    sorted_labels = [scale_labels[index] for index in fifth_indices]

    if shift:
        sorted_features = [
            np.roll(feature, shift=shift, axis=0)
            for feature in sorted_features
        ]
        sorted_labels = _roll_sequence(sorted_labels, shift)

    if scale_features_is_sequence:
        return sorted_features, sorted_labels

    return sorted_features[0], sorted_labels


def _get_fifths_order(
    scale_labels: Sequence[str],
    *,
    global_key: Optional[str],
) -> Tuple[np.ndarray, int]:
    """Return fifths-order row indices and the shift needed for centering."""

    fifth_indices = np.mod(np.arange(12) * 7, 12)
    if len(scale_labels) != 12:
        raise ValueError("scale labels must contain exactly 12 entries for fifths sorting")

    if global_key is None:
        return fifth_indices, 0

    sorted_labels = [scale_labels[index] for index in fifth_indices]
    center_index = len(sorted_labels) // 2
    key_index = _find_global_key_index(
        sorted_labels,
        fifth_indices=fifth_indices,
        global_key=global_key,
    )
    return fifth_indices, center_index - key_index


def _find_global_key_index(
    labels: Sequence[str],
    *,
    fifth_indices: np.ndarray,
    global_key: str,
) -> int:
    """Find the fifths-sorted row index corresponding to ``global_key``."""

    normalized_label = _normalize_label(global_key)
    for index, candidate in enumerate(labels):
        if _normalize_label(candidate) == normalized_label:
            return index

    try:
        pitch_class = key_label_to_pitch_class(global_key)
    except ValueError:
        pitch_class = None
    if pitch_class is not None:
        matches = np.flatnonzero(fifth_indices == pitch_class)
        if matches.size:
            return int(matches[0])

    raise ValueError(f'global_key "{global_key}" is not present in the scale labels')


def _normalize_label(label: str) -> str:
    """Normalize a key or scale label for robust string comparison."""

    label = label.strip().lower().replace("-", "b")
    return label.split(":", maxsplit=1)[0]


def _roll_sequence(sequence: Sequence[str], shift: int) -> list[str]:
    """Roll a sequence by the same convention as ``numpy.roll``."""

    values = list(sequence)
    if not values:
        return values

    shift = shift % len(values)
    return values[-shift:] + values[:-shift]
