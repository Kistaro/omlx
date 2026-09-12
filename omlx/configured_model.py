# SPDX-License-Identifier: Apache-2.0
"""Centralized resolution for model configurations.

In general, model settings override model defaults, and model defaults
override sampling defaults. Settings not related to sampling are expected
to be absent from sampling defaults, and not all of these settings objects
use the same name for the same concept.

This object is intended to be incrementally extended and incrementally
adopted: when updating code that performs this kind of resolution, migrate
it to use this object if practical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from .engine_pool import EngineEntry
from .model_settings import ModelSettings, merge_chat_template_kwargs

if TYPE_CHECKING:
    # SamplingDefaults lives in server.py, which imports this module.
    from .server import SamplingDefaults

T = TypeVar("T")


def first_present(*args: T | None) -> T | None:
    """The first non-None value passed, or None if all are None."""
    for x in args:
        if x is not None:
            return x
    return None


def _empty_engine_entry() -> EngineEntry:
    """A placeholder entry for an unknown model: every default is ``None``."""
    return EngineEntry(
        model_id="",
        model_path="",
        model_type="llm",
        engine_type="batched",
        estimated_size=0,
    )


@dataclass(frozen=True)
class ConfiguredModel:
    """A model, its configuration, and its sources for fallback settings.

    Every layer is always present. Layers with no real data use default-
    constructed instances (e.g. ``ModelSettings()`` with every field
    ``None``, an ``EngineEntry`` with empty strings and zeroes, etc.).
    This keeps property accessors simple -- no ``None`` guards needed.
    However, multilayer resolution will not continue past a field that
    has a default value; currently this does not affect any lookups
    but an alternative approach may be required in the future.
    """

    # Priority is generally settings > model_entry > sampling.

    settings: ModelSettings
    model_entry: EngineEntry
    sampling: SamplingDefaults

    def _settings_template_kwargs(self) -> dict[str, Any]:
        """Chat-template kwargs the settings layer sends absent request overrides.

        Delegates to the same helper the request path uses, so the reported
        state cannot drift from what is actually rendered.
        """
        return merge_chat_template_kwargs(
            self.settings,
            None,
            preserve_thinking_default=self.model_entry.preserve_thinking_default,
        )

    @property
    def enable_thinking(self) -> bool | None:
        """Effective thinking state as the engine will serve it.

        Resolved the way the request path renders it: the dedicated
        ``settings.enable_thinking`` toggle, else ``settings.chat_template_kwargs``,
        else an active thinking budget (which switches thinking on), else the
        model's chat-template default (``model_entry.thinking_default``).
        ``None`` means the model exposes no thinking toggle.
        """
        return first_present(
            self._settings_template_kwargs().get("enable_thinking"),
            self.model_entry.thinking_default,
        )

    @property
    def preserve_thinking(self) -> bool | None:
        """Effective ``preserve_thinking`` state (keep <think> blocks in
        historical turns).

        Follows the request path: the dedicated toggle, else
        ``settings.chat_template_kwargs``, else ``True`` when the template
        supports the flag (``model_entry.preserve_thinking_default``) and
        thinking is not switched off. ``None`` when the template has no such
        flag.
        """
        return self._settings_template_kwargs().get("preserve_thinking")

    @property
    def max_context_window(self) -> int | None:
        """Effective max context window limit.

        Resolution:
            1. **Per-model override** (admin UI / settings.json) -- always
               wins. An operator who has set a per-model number knows what
               they want; ``max_context_window_policy`` does not clamp it.
            2. **Model-config-discovered native context length** (#1308),
               optionally clamped by the operator policy: if
               ``sampling.max_context_window_policy`` is set, use
               ``min(native, policy)``; otherwise use ``native`` as-is.
            3. **Fallback default** from ``sampling.max_context_window`` --
               only used when neither tier 1 nor tier 2 yields a value.
               Treated as a default, NOT capped by the policy; existing
               ``settings.json`` files carrying the historical ``32768``
               default keep working unchanged after upgrade.

        The policy field is intentionally nullable and unset by default so
        no existing install behavior shifts. Setting it engages
        ``min(native, policy)`` across every model whose native context is
        discoverable; per-model overrides remain the operator's escape
        hatch for individual models that should exceed the policy.
        """
        if self.settings.max_context_window is not None:
            return self.settings.max_context_window
        native = self.model_entry.model_context_length
        policy = self.sampling.max_context_window_policy
        if native is not None and policy is not None and policy > 0:
            native = min(native, policy)
        return first_present(native, self.sampling.max_context_window)

    @property
    def max_tokens(self) -> int | None:
        """Effective max output tokens."""
        return first_present(
            self.settings.max_tokens,
            self.sampling.max_tokens,
        )


def new_configured_model(
    settings: ModelSettings | None = None,
    model_entry: EngineEntry | None = None,
    sampling: SamplingDefaults | None = None,
) -> ConfiguredModel:
    """Build a ConfiguredModel, filling absent layers with defaults."""
    # Lazy import: SamplingDefaults lives in server.py, which imports us.
    from .server import SamplingDefaults

    return ConfiguredModel(
        settings=ModelSettings() if settings is None else settings,
        model_entry=_empty_engine_entry() if model_entry is None else model_entry,
        sampling=SamplingDefaults() if sampling is None else sampling,
    )
