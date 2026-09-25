"""Test package marker.

Required so `tests.conftest` resolves to this directory. Without it, Python
finds an unrelated `tests` package installed in site-packages first and the
factory helpers import from the wrong place.
"""
