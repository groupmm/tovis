"""Scale and local-key analysis utilities for TOVIS."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import key_label_to_pitch_class, normalize_feature_sequence


ScaleLabelVersion = str
TemplateVersion = str
TemplateAnalysisMethod = str
NormType = str


def get_scale_labels(
    version: ScaleLabelVersion = "absolute_diatonic",
    *,
    global_key: str = "C",
) -> list[str]:
    """Return labels for the 12 major-scale templates.

    Parameters
    ----------
    version:
        Label set. Options are ``"absolute_diatonic"``, which returns
        circle-of-fifths labels relative to C such as ``"0"``, ``"1b"``, and
        ``"1#"``; and ``"relative_diatonic"``, which returns signed
        circle-of-fifths distances relative to ``global_key`` such as ``"0"``,
        ``"+1"``, and ``"-1"``.
    global_key:
        Key used as the zero point for ``version="relative_diatonic"``.
        Ignored by ``version="absolute_diatonic"``.

    Returns
    -------
    list of str
        Twelve labels ordered by chromatic pitch class from C to B.
    """

    if version == "absolute_diatonic":
        return [_format_absolute_diatonic_label(_fifths_distance_from_c(pitch_class)) for pitch_class in range(12)]
    if version == "relative_diatonic":
        global_pitch_class = key_label_to_pitch_class(global_key)
        return [
            _format_relative_diatonic_label(
                _fifths_distance_between(global_pitch_class, pitch_class)
            )
            for pitch_class in range(12)
        ]

    raise ValueError('version must be "absolute_diatonic" or "relative_diatonic"')


def _fifths_distance_from_c(pitch_class: int) -> int:
    """Return signed circle-of-fifths distance from C to ``pitch_class``."""

    return _fifths_distance_between(0, pitch_class)


def _fifths_distance_between(reference_pitch_class: int, pitch_class: int) -> int:
    """Return signed fifths distance from a reference pitch class."""

    distance = ((pitch_class - reference_pitch_class) * 7) % 12
    if distance > 5:
        distance -= 12
    return int(distance)


def _format_absolute_diatonic_label(distance: int) -> str:
    """Format a signed fifths distance as an absolute diatonic label."""

    if distance == 0:
        return "0"
    if distance > 0:
        return f"{distance}#"
    return f"{abs(distance)}b"


def _format_relative_diatonic_label(distance: int) -> str:
    """Format a signed fifths distance as a relative diatonic label."""

    if distance == 0:
        return "0"
    if distance > 0:
        return f"+{distance}"
    return str(distance)


def get_templates(version: TemplateVersion = "triads_enhanced") -> np.ndarray:
    """Return a root-position major-scale template.

    Parameters
    ----------
    version:
        Template weighting scheme. ``"binary"`` marks diatonic pitch classes,
        ``"triads_enhanced"`` emphasizes tonic-triad tones, and
        ``"krumhansl"`` uses a perceptual weighting.

    Returns
    -------
    numpy.ndarray
        Column vector of shape ``(12, 1)`` for the C major template.
    """

    if version == "binary":
        return np.array([[1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]], dtype=float).T
    if version == "triads_enhanced":
        return np.array([[3, 0, 1, 0, 3, 1, 0, 2, 0, 2, 0, 1]], dtype=float).T
    if version == "krumhansl":
        return np.array(
            [[2.9663, 0, 1.3762, 0, 2.6413, 1.5088, 0, 2.0663, 0, 2.2488, 0, 1.3000]],
            dtype=float,
        ).T

    raise ValueError('version must be "binary", "triads_enhanced", or "krumhansl"')


def generate_template_matrix(templates: np.ndarray) -> np.ndarray:
    """Generate a template matrix by cyclically shifting 12-bin templates.

    Every input template is transposed through all twelve chromatic roots.

    Parameters
    ----------
    templates:
        Template vector or matrix of shape ``(12,)`` or ``(12, n_templates)``.

    Returns
    -------
    template_matrix:
        Matrix of shape ``(12, 12 * n_templates)``. For a single major-scale
        template, columns correspond to C, Db, D, ..., B.
    """

    templates = np.asarray(templates, dtype=float)
    if templates.ndim == 1:
        templates = templates[:, np.newaxis]
    if templates.ndim != 2 or templates.shape[0] != 12:
        raise ValueError("templates must have shape (12,) or (12, n_templates)")

    template_matrix = np.zeros((12, 12 * templates.shape[1]))
    for shift in range(12):
        template_matrix[:, shift::12] = np.roll(templates, shift, axis=0)

    return template_matrix


def scale_analysis_templates(
    chromagram: np.ndarray,
    templates: Optional[np.ndarray] = None,
    *,
    template_version: Optional[TemplateVersion] = None,
    method: TemplateAnalysisMethod = "exp_product",
    apply_output_normalization: bool = True,
    norm: NormType = "2",
    norm_threshold: float = 1e-12,
    chroma_floor: Optional[float] = 1e-3,
    apply_softmax_rescaling: bool = False,
    softmax_eta: float = 1.0,
) -> np.ndarray:
    r"""Estimate scale likelihoods from a chromagram using scale templates.

    The chromagram frames and template vectors are L2-normalized before
    comparison. With ``method="inner_product"``, the likelihood is the
    inner product :math:`L_k(n)=t_k^\mathsf{T}x_n`. With
    ``method="exp_product"``, it is the exponent-weighted product
    :math:`L_k(n)=\prod_p x_n(p)^{t_k(p)}`.

    Parameters
    ----------
    chromagram:
        Chromagram of shape ``(12, n_frames)``.
    templates:
        Template matrix of shape ``(12, n_templates)``. Required unless
        ``template_version`` is supplied.
    template_version:
        Optional built-in template version. Options are ``"binary"``,
        ``"triads_enhanced"``, and ``"krumhansl"``. If given, a
        twelve-transposition template matrix is generated internally and
        ``templates`` is ignored.
    method:
        Template comparison method. Options are ``"inner_product"`` and
        ``"exp_product"`` as defined above.
    apply_output_normalization:
        If true, normalize each output frame.
    norm:
        Output frame normalization when ``apply_output_normalization`` is
        true. Options are ``"1"``, ``"2"``, ``"max"``, ``"z"``, and
        ``None``; see :func:`~tovis.utils.normalize_feature_sequence`.
    norm_threshold:
        Threshold passed to :func:`~tovis.utils.normalize_feature_sequence`
        for output normalization.
    chroma_floor:
        Optional minimum chroma value used by the ``"exp_product"`` method after
        input normalization: :math:`x\leftarrow\max(x,\mathtt{chroma\_floor})`.
        This prevents hard zeros from collapsing complete likelihood frames.
    apply_softmax_rescaling:
        If true, apply softmax rescaling to the output after optional
        normalization.
    softmax_eta:
        Softmax sharpness parameter :math:`\eta` in
        :math:`\operatorname{softmax}(\eta L)`.

    Returns
    -------
    scale_likelihood:
        Scale likelihood matrix of shape ``(n_templates, n_frames)``.
    """

    chromagram = np.asarray(chromagram, dtype=float)
    if template_version is not None:
        templates = generate_template_matrix(get_templates(template_version))
    elif templates is None:
        raise ValueError("templates or template_version must be provided")
    else:
        templates = np.asarray(templates, dtype=float)

    if chromagram.ndim != 2 or chromagram.shape[0] != 12:
        raise ValueError("chromagram must have shape (12, n_frames)")
    if templates.ndim != 2 or templates.shape[0] != 12:
        raise ValueError("templates must have shape (12, n_templates)")

    chromagram_normalized = normalize_feature_sequence(chromagram, norm="2")
    templates_normalized = normalize_feature_sequence(templates, norm="2")

    if method == "inner_product":
        scale_likelihood = templates_normalized.T @ chromagram_normalized
    elif method == "exp_product":
        if chroma_floor is not None:
            if chroma_floor < 0:
                raise ValueError("chroma_floor must be non-negative")
            chromagram_normalized = np.clip(
                chromagram_normalized,
                a_min=chroma_floor,
                a_max=None,
            )

        scale_likelihood = np.prod(
            np.power(
                chromagram_normalized[np.newaxis, :, :],
                templates_normalized.T[:, :, np.newaxis],
            ),
            axis=1,
        )
    else:
        raise ValueError('method must be "inner_product" or "exp_product"')

    if apply_output_normalization:
        scale_likelihood = normalize_feature_sequence(
            scale_likelihood,
            norm=norm,
            threshold=norm_threshold,
        )

    if apply_softmax_rescaling:
        scale_likelihood = softmax_rescaling(
            scale_likelihood,
            eta=softmax_eta,
            axis=0,
        )

    return scale_likelihood


def softmax_rescaling(
    x: np.ndarray,
    *,
    eta: float = 1.0,
    axis: int = 0,
) -> np.ndarray:
    r"""Apply numerically stable softmax rescaling.

    Computes :math:`y_i=\exp(\eta x_i)/\sum_j\exp(\eta x_j)` along ``axis``.

    Parameters
    ----------
    x:
        Input array.
    eta:
        Sharpness factor :math:`\eta`. Larger positive values concentrate mass
        around larger inputs; ``0`` produces a uniform distribution.
    axis:
        Axis along which the softmax is computed.

    Returns
    -------
    numpy.ndarray
        Softmax-normalized array with the same shape as ``x``.
    """

    x = np.asarray(x, dtype=float)
    x_scaled = eta * x
    x_scaled = x_scaled - np.max(x_scaled, axis=axis, keepdims=True)
    x_exp = np.exp(x_scaled)
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)
