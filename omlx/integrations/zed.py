"""Zed editor integration."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from omlx.integrations.base import Integration, IntegrationContext


class ZedIntegration(Integration):
    """Zed editor integration that configures all oMLX models."""

    CONFIG_PATH = Path.home() / ".config" / "zed" / "settings.json"

    def __init__(self):
        super().__init__(
            name="zed",
            display_name="Zed",
            type="config_file",
            install_check="zed",
            install_hint="Install from https://zed.dev",
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        return f"omlx launch zed --model {ctx.model or 'select-a-model'}"

    @staticmethod
    def _is_reasoning_model(model: str | None) -> bool:
        """Detect reasoning/thinking models by name patterns."""
        if not model:
            return False
        lower = model.lower()
        # Exclude explicit non-thinking variants first
        if "no-thinking" in lower or "nothinking" in lower:
            return False
        # Common reasoning model patterns
        return bool(
            re.search(
                r"\b(o[13]|o\d+|r1|deepseek[_-]?r|gemini[_-]?2\.5[_-]?pro[_-]?thinking|"
                r"gpt[_-]?o[13]|claude[_-]?sonnet[_-]?4[_-]?thinking|"
                r"qwen3[.-]?[56]?[.-]?[0-9]+[.-]?thinking|"
                r"qwen[.-]?2[.-]?[0-9]+[.-]?[mgb]?[.-]?thinking|"
                r"[tT]hinking)\b",
                lower,
            )
        )

    def _build_model_entry(self, model_info: dict) -> dict:
        """Build a Zed model entry from oMLX model info.

        Args:
            model_info: Model info dict from IntegrationContext.all_models
                        (has id, enable_thinking, max_context_window, max_tokens, model_type, etc.)
        """
        model_id = model_info["id"]
        entry: dict = {"name": model_id}

        # Context window (Zed calls this max_tokens)
        context_window = model_info.get("max_context_window")
        if context_window:
            entry["max_tokens"] = context_window

        # Output token limit
        max_tokens = model_info.get("max_tokens")
        if max_tokens:
            entry["max_output_tokens"] = max_tokens

        # Capabilities (defaults per Zed OpenAI-compatible spec)
        capabilities: dict = {
            "tools": True,
            "images": model_info.get("model_type") == "vlm",
            "parallel_tool_calls": False,
            "prompt_cache_key": False,
            "chat_completions": True,
            "interleaved_reasoning": False,
        }

        # Reasoning detection
        enable_thinking = model_info.get("enable_thinking", False)
        # Fallback: detect from model name
        if not enable_thinking:
            enable_thinking = self._is_reasoning_model(model_id)

        if enable_thinking:
            capabilities["interleaved_reasoning"] = True
            entry["reasoning_effort"] = "high"

        entry["capabilities"] = capabilities
        return entry

    def configure(self, ctx: IntegrationContext) -> None:
        """Configure Zed with all oMLX models."""
        all_models = ctx.all_models
        if not all_models:
            print("Warning: no models found from oMLX server")
            return

        def update_settings(config: dict) -> None:
            # Set up language_models.openai_compatible.oMLX
            config.setdefault("language_models", {})
            config["language_models"].setdefault("openai_compatible", {})

            # Build available models list
            available_models = [self._build_model_entry(m) for m in all_models]

            provider_config: dict = {
                "api_url": ctx.openai_base_url,
                "available_models": available_models,
            }
            config["language_models"]["openai_compatible"]["oMLX"] = provider_config

            # Set default model if one is selected
            if ctx.model:
                config.setdefault("agent", {})
                # Detect if the selected model supports reasoning
                model_status = ctx.all_models_status.get(ctx.model, {})
                enable_thinking = model_status.get("enable_thinking", False)
                if not enable_thinking:
                    enable_thinking = self._is_reasoning_model(ctx.model)

                config["agent"]["default_model"] = {
                    "provider": "oMLX",
                    "model": ctx.model,
                    "enable_thinking": bool(enable_thinking),
                }

        self._write_jsonc_config(self.CONFIG_PATH, update_settings)

    def launch(self, ctx: IntegrationContext) -> None:
        """Configure and launch Zed."""
        self.configure(ctx)

        # Verify zed is available before attempting to exec
        if not shutil.which("zed"):
            print(
                "zed not found on PATH. Install from https://zed.dev "
                "and make sure it is available in your shell."
            )
            raise SystemExit(1)

        # Set API key as environment variable (Zed reads <PROVIDER>_API_KEY)
        env = self._scrubbed_env()
        env["OMLX_API_KEY"] = ctx.auth_token

        # Launch Zed
        os.execvpe("zed", ["zed"], env)
