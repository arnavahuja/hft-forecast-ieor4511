"""Short-horizon forecasting: labels, features, purged CV, models and evaluation.

Sits between ``analytics`` and ``services`` in the layer order: ``ml.economic`` drives the
engine and ``ml.evaluate`` uses ``analytics.stats``, while ``services`` orchestrates ``ml``.
Models and feature sets get arrays in and arrays out and may not import the rest of ``ml``.
"""
