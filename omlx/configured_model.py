"""A configured model: a model paired with its per-model configuration.

``ConfiguredModel`` is the small object that holds the two halves the server
already shuffles around separately — an :class:`~omlx.engine_pool.EngineEntry`
(the *model*: what its chat template defaults to, its declared context length,
etc.) and a :class:`~omlx.model_settings.ModelSettings` (the *configuration*:
the user's explicit overrides) — and owns the handful of resolutions that
depend on *exactly* those two inputs.

Deliberately **not** here: anything that also consults the global
``SamplingDefaults`` tier (``max_tokens``, ``max_context_window``, sampling
params). Those are a three-input resolution and live in ``get_max_context_window``
/ ``get_resolved_sampling_params`` in ``server.py``. Keeping this object to the
two-input resolutions is what keeps it two fields; if a method here needed a
third input, the abstraction would be lying.

The reasoning/thinking resolution is the motivating case (see #?, the Pi
integration was guessing "is this a reasoning model" from the model slug
instead of reading the model's configured thinking state). Both the
``/v1/models/status`` endpoint (which reports the *effective* state to
integration clients) and the chat-completion handlers (which inject template
kwargs) source that logic from here so they cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from .model_settings import ModelSettings

if TYPE_CHECKING:  # avoid importing the heavy engine_pool module at runtime
    from .engine_pool import EngineEntry


@dataclass(frozen=True)
class ConfiguredModel:
    """A :class:`ModelSettings` (configuration) bound to an :class:`EngineEntry`
    (model).

    ``entry`` is optional: the chat-completion handlers only need the *override*
    accessors (which read settings alone), so they can construct
    ``ConfiguredModel(ms)`` without paying for an entry lookup. The *resolved*
    accessors fall back to ``None`` when no entry is present.
    """

    settings: ModelSettings
    entry: Optional["EngineEntry"] = None

    # ------------------------------------------------------------------ #
    # Resolved effective state — per-model override wins, else the model's
    # own template default. Tri-state: True/False, or None when the model has
    # no thinking toggle at all. This is what we *report* (e.g. to integration
    # clients via /v1/models/status), not what we force into the template.
    # ------------------------------------------------------------------ #

    @property
    def enable_thinking(self) -> Optional[bool]:
        """Effective thinking state as the engine will serve it.

        ``settings.enable_thinking`` (the explicit toggle from Model settings)
        takes precedence; otherwise the model's chat-template default
        (``entry.thinking_default``). ``None`` means the model exposes no
        thinking toggle.
        """
        if self.settings.enable_thinking is not None:
            return self.settings.enable_thinking
        return getattr(self.entry, "thinking_default", None)

    @property
    def preserve_thinking(self) -> Optional[bool]:
        """Effective ``preserve_thinking`` state (keep <think> blocks in
        historical turns), resolved the same way as :attr:`enable_thinking`.
        """
        if self.settings.preserve_thinking is not None:
            return self.settings.preserve_thinking
        return getattr(self.entry, "preserve_thinking_default", None)

    # ------------------------------------------------------------------ #
    # Inference-time contract — inject a template kwarg only when the user set
    # an explicit value, leaving the template free to apply its own default
    # otherwise. Intentionally does NOT fold in the template default (that's
    # what the resolved accessors above are for).
    # ------------------------------------------------------------------ #

    def thinking_template_overrides(self) -> Dict[str, Any]:
        """Chat-template kwargs to merge from the explicit per-model toggles.

        Returns only the keys whose override is set (not ``None``), so an unset
        toggle defers to the template's built-in default — preserving the
        long-standing behaviour of the chat-completion handlers.
        """
        overrides: Dict[str, Any] = {}
        if self.settings.enable_thinking is not None:
            overrides["enable_thinking"] = self.settings.enable_thinking
        if self.settings.preserve_thinking is not None:
            overrides["preserve_thinking"] = self.settings.preserve_thinking
        return overrides
