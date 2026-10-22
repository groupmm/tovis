# Third-party notices

TOVIS is developed by Johannes Zeitler and Meinard Müller at International
Audio Laboratories, Erlangen. The root MIT license applies to TOVIS-authored
code. Bundled third-party code retains its upstream notices and terms.

## Onsets & Frames

The code in `tovis/transcription/onsets_and_frames/` derives from the PyTorch
implementation of Onsets & Frames by Jong Wook Kim and the CountEM codebase.

- Original implementation: https://github.com/jongwook/onsets-and-frames
- Copyright 2019 Jong Wook Kim; MIT license, reproduced in
  `licenses/onsets-and-frames-MIT.txt`.
- The underlying Onsets and Frames architecture was introduced by Hawthorne
  et al., *Onsets and Frames: Dual-Objective Piano Transcription* (2018).

## CountEM / Count the Notes

TOVIS includes code from the count-the-notes project (CountEM), including the
`CountEMModel` Hugging Face wrapper in `hf_model.py` and extensions to the
Onsets & Frames implementation.

- Source: https://github.com/Yoni-Yaffe/count-the-notes
- Authors: Jonathan Yaffe, Ben Maman, Meinard Müller, and Amit Bermano (2025).
- Project: *Count the Notes: Histogram-Based Supervision for Automatic Music
  Transcription*.
- Upstream license notice: `licenses/countem-upstream.txt`, reproduced as
  published in the upstream `LICENSE.md`.

Model weights are not included in the TOVIS distributions.

## PyTorch STFT

The STFT implementation in `tovis/transcription/onsets_and_frames/mel.py`
is adapted from Prem Seetharaman's PyTorch STFT implementation.

- Source: https://github.com/pseeth/pytorch-stft (now `pseeth/torch-stft`).
- Copyright (c) 2017, Prem Seetharaman; BSD 3-Clause license, reproduced in
  `licenses/pytorch-stft-BSD-3-Clause.txt`.

TOVIS integrates these components into its package and transcription wrapper;
the bundled copies may differ from upstream. Existing upstream model metadata
and source credits are retained.
