"""Regression checks for the public API and feature time origins."""

import unittest

import numpy as np
import pandas as pd

import tovis


class PublicAPITest(unittest.TestCase):
    def test_wildcard_import_resolves_every_export(self):
        namespace = {}
        exec("from tovis import *", namespace)
        self.assertTrue(all(name in namespace for name in tovis.__all__))

    def test_default_feature_time_origin_is_zero(self):
        sync = pd.DataFrame({"time": [0.0, 1.0, 2.0], "measure": [1.0, 2.0, 3.0]})
        features = np.array([[10.0, 20.0, 30.0]])
        converted, measures = tovis.time_to_measures(
            features, sync, feature_rate=1.0, frames_per_measure=1
        )
        np.testing.assert_allclose(measures, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(converted, features)

    def test_explicit_feature_time_origin_is_respected(self):
        sync = pd.DataFrame({"time": [0.0, 1.0, 2.0], "measure": [1.0, 2.0, 3.0]})
        features = np.array([[10.0, 20.0]])
        converted, measures = tovis.time_to_measures(
            features, sync, feature_rate=1.0, time_start=1.0, frames_per_measure=1
        )
        np.testing.assert_allclose(measures, [2.0, 3.0])
        np.testing.assert_allclose(converted, features)


if __name__ == "__main__":
    unittest.main()
