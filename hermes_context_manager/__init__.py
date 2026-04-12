"""Hermes Context Manager — silent-first context optimization plugin.

The Hermes plugin loader entry point lives in the REPO-ROOT ``__init__.py``
(not here).  This file exports the plugin class for direct imports from
tests and standalone usage only.
"""

from .plugin import HermesContextManagerPlugin

__all__ = ["HermesContextManagerPlugin"]
