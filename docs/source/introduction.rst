Introduction
============

TOVIS turns musical audio or note annotations into chroma features, compares
those features with scale templates, and visualizes tonal and formal structure.

A runnable example
------------------

Install TOVIS as described in :doc:`installation`, then run the following code
in Python. It synthesizes a three-second C-major chord, extracts an STFT
chromagram, computes scale likelihoods, and saves ``tovis-introduction.png``
in the current directory. No datasets, audio downloads, or model checkpoints
are needed. Saving the plot also works without a graphical desktop.

.. literalinclude:: _examples/introduction.py
   :language: python

Reading the results
-------------------

``chroma`` has shape ``(12, n_frames)``. Its rows represent pitch classes
ordered C, C-sharp, D, ..., B; columns represent successive audio frames.
``feature_rate`` is measured in frames per second, so frame ``n`` corresponds
to time ``n / feature_rate``. A C-major chord emphasizes C, E, and G.

``likelihood`` compares each chroma frame with twelve transposed major-scale
templates. The default scale labels describe each scale's position on the
circle of fifths relative to C. These values describe template agreement;
they are not calibrated probabilities or a definitive key estimate.

Use your own audio
------------------

Replace the synthesis and chromagram extraction in the example with:

.. code-block:: python

   chroma, feature_rate = tovis.compute_chromagram_from_file("your-audio.wav")

The default backend is CQT. STFT, CQT, VQT, and IIR are available without
neural models. See :doc:`installation` for the additional requirements of
the neural backends.

Use note annotations
--------------------

For symbolic input, provide note start and end times in seconds and MIDI
pitch numbers in a DataFrame:

.. code-block:: python

   import pandas as pd

   notes = pd.DataFrame({
       "start": [0.0, 0.0, 0.0],
       "end": [3.0, 3.0, 3.0],
       "pitch": [60, 64, 67],
   })
   chroma, feature_rate = tovis.compute_chromagram_from_annotations(
       notes, feature_rate=10.0
   )

This returns a ``(12, 30)`` matrix at ten frames per second. It can be used
with the same scale-analysis and visualization functions. See :doc:`api`
for smoothing, normalization, synchronization, and structure-analysis options.
