"""Registry for custom-mode validators."""

from __future__ import annotations

from rmv.plugins.base import CustomModeValidator

_PLUGINS: dict[str, CustomModeValidator] = {}


def register(plugin: CustomModeValidator) -> None:
    """Register a custom mode validator by mode_id."""
    _PLUGINS[plugin.mode_id] = plugin


def get(mode_id: str) -> CustomModeValidator | None:
    """Return plugin for mode_id, or None if not registered."""
    return _PLUGINS.get(mode_id)


def list_plugins() -> list[str]:
    """Return registered mode_id values."""
    return sorted(_PLUGINS.keys())


def get_plugin_description(mode_id: str) -> str | None:
    plugin = get(mode_id)
    if plugin is None:
        return None
    return plugin.description


def _register_builtin_plugins() -> None:
    from rmv.plugins import sleipnir_8qpsk

    register(sleipnir_8qpsk.Sleipnir8QPSKValidator())


_register_builtin_plugins()
