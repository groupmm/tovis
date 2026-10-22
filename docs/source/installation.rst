Installation
============

TOVIS requires Python 3.11 or newer. Its Python package and distribution name
is ``tovis``. Use a virtual environment or the supplied conda environment to
keep dependencies separate from other projects.

Install from PyPI
-----------------

.. code-block:: bash

   python -m pip install tovis

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/groupmm/tovis.git
   cd tovis
   python -m pip install .

Check the installation with:

.. code-block:: bash

   python -c "import tovis; from importlib.metadata import version; print(version('tovis'))"

Optional dependencies
---------------------

Install optional dependencies for documentation, notebooks, or neural
transcription with:

.. code-block:: bash

   python -m pip install "tovis[docs]"
   python -m pip install "tovis[notebooks]"
   python -m pip install "tovis[transcription]"

``docs`` installs Sphinx and the documentation theme. ``notebooks`` installs
JupyterLab. ``transcription`` installs PyTorch for neural inference with
user-supplied model checkpoints. Multiple
extras can be combined, for example ``"tovis[docs,notebooks]"``. For a source
checkout, replace ``tovis`` with ``.`` in these commands.

The STFT, CQT, VQT, and IIR chroma backends need no neural checkpoints or
PyTorch. Neural backends additionally require a compatible checkpoint, which
is obtained separately and not included in the distribution.
You must configure the checkpoint path explicitly for every
neural transcription call. TOVIS searches no default checkpoint location.
For the feature-extraction functions, set ``transcription_checkpoint``:

.. code-block:: python

   from tovis import compute_chromagram_from_file

   chroma, feature_rate = compute_chromagram_from_file(
       "your-audio.wav",
       backend="ONSETS_AND_FRAMES",
       transcription_checkpoint="/path/to/checkpoint_OnsetsAndFrames.pt",
       transcription_device="cpu",
   )

For Deep Chroma use ``backend="DEEP_CHROMA"`` and a compatible Deep Chroma
checkpoint. Direct calls to functions in ``tovis.transcription`` require the
``checkpoint`` keyword argument. Missing or empty paths are rejected, and
paths must point to existing files before audio loading or neural inference.

These backends expect different model formats; arbitrary upstream
weights are not interchangeable. Use checkpoints from a trusted source:
the Onsets & Frames loader deserializes a Python model object.

Development and documentation
-----------------------------

For editable installation and a local documentation build:

.. code-block:: bash

   python -m pip install -e ".[docs,notebooks]"
   python -m unittest discover -s tests
   python -m sphinx -W -b html docs/source docs/build/html

Open ``docs/build/html/index.html`` in a browser. Generated documentation is
excluded from version control. The conda environment in ``environment.yml``
is an alternative for local development; install TOVIS in that environment
with ``python -m pip install -e .``.

The demonstration notebooks and their example audio and annotation data are
in ``scripts/`` in the source repository. Neural examples additionally require
compatible checkpoints supplied by the user. The :doc:`introduction` provides
a self-contained example using synthesized audio.
