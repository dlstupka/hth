"""Regression-shape coverage is intentionally consolidated in active unittest modules.

The former pytest-style free functions in this file were never executed by the
repository's unittest contract and duplicated or contradicted canonical coverage
in test_optimizer_intelligence, test_preferred_dispatch, and test_parallelism_store.
"""
