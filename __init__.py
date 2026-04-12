"""Hermes plugin entrypoint for Hermes Context Manager."""

from __future__ import annotations

from pathlib import Path

# When loaded by the Hermes plugin loader, this module is
# hermes_plugins.hermes_context_manager and the inner package is a
# relative submodule.  When loaded directly (tests, standalone), the
# inner package is a top-level absolute import.
try:
    from .hermes_context_manager.plugin import HermesContextManagerPlugin
except ImportError:
    from hermes_context_manager.plugin import HermesContextManagerPlugin

_PLUGIN = HermesContextManagerPlugin(plugin_dir=Path(__file__).resolve().parent)


def register(ctx) -> None:
    """Register the plugin with Hermes."""
    _PLUGIN.register(ctx)
