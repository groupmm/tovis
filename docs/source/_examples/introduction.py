import matplotlib.pyplot as plt
import numpy as np

import tovis
from tovis.visualization import plot_chromagram, plot_scale_likelihood

sample_rate = 22050
time = np.arange(3 * sample_rate) / sample_rate
frequencies = [261.63, 329.63, 392.00]  # C4, E4, G4
audio = sum(np.sin(2 * np.pi * f * time) for f in frequencies) / 3

chroma, feature_rate = tovis.compute_chromagram(
    audio, sample_rate=sample_rate, backend="STFT"
)
likelihood = tovis.scale_analysis_templates(chroma, template_version="binary")

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
plot_chromagram(chroma, feature_rate=feature_rate, ax=axes[0])
plot_scale_likelihood(likelihood, feature_rate=feature_rate, ax=axes[1])
fig.tight_layout()
fig.savefig("tovis-introduction.png", dpi=150)
plt.close(fig)
