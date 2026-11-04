"""Cline CLI with its `version` option installing exactly that release.

Harbor 0.25.0's cline-cli adapter accepts the shared `version` option, and
reports it as the harness version, but installs `cline@nightly` unless its own
`cline_version` option is set. The bench pins every harness through `version`
(reuse, the ledger, the reports and pre-baking all read it), so this subclass
makes `version` set `cline_version` too.

It also passes `-P` in Cline's provider names where they differ from
LiteLLM's: Harbor sends the LiteLLM name as it is, but Cline 3.0.74 knows
Fireworks as `fireworks` and OpenAI's own API as `openai-native` (its
`openai` is a generic OpenAI-compatible endpoint that needs a base URL). The
key still reaches it through `-k`, from the provider's own variable. xAI is
`xai` to both. Everything else is Harbor's ClineCli unchanged.

The cline arms use it as `agent: bench.agents.cline:PinnedClineCli` with
`family: cline-cli`.
"""

from __future__ import annotations

from typing import Any

from harbor.agents.installed.cline import ClineCli

# LiteLLM provider name -> Cline's (@cline/llms 0.2.0's provider ids).
CLINE_PROVIDERS = {"fireworks_ai": "fireworks", "openai": "openai-native"}


class PinnedClineCli(ClineCli):
    def __init__(self, *args: Any, version: str | None = None, cline_version: str | None = None,
                 **kwargs: Any) -> None:
        if version and cline_version and str(version) != str(cline_version):
            raise ValueError(f"cline-cli: version {version!r} and cline_version {cline_version!r} differ; "
                             "set version only")
        super().__init__(*args, version=version, cline_version=cline_version or version, **kwargs)

    def _parse_model_name(self, model_name: str) -> tuple[str | None, str]:
        """The run command's provider in Cline's spelling. Only the command uses
        this; the model connection (key, endpoint, the trial's recorded provider)
        keeps LiteLLM's name."""
        provider, model = super()._parse_model_name(model_name)
        return CLINE_PROVIDERS.get(provider or "", provider), model
