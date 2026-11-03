"""
Base classes for compaction passes.

Uses LLMConfigManager for all LLM calls, forcing the provider and model of the
active compaction profile: Groq gpt-oss-120b by default (architecture decision
D9), see resolve_compaction_profile().
"""
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

logger = logging.getLogger("synap.cloud.compaction.passes")

from synap.cloud.infrastructure.llm import LLMConfigManager, LLMProvider
from synap.cloud.infrastructure.llm import settings as llm_settings
from synap.cloud.shared import edition

from ..models import (
    PassResult,
    CompactionConfig,
    ConversationMessage
)
from ..interfaces import ICompactionPass


@dataclass(frozen=True)
class CompactionProfile:
    """Which model writes a compaction, and what runs around it."""
    name: str
    provider: LLMProvider
    model: str
    # Added to max_tokens. Groq's gpt-oss models are reasoning models and spend
    # reasoning tokens out of that budget (up to 1,949 output tokens on one
    # pass call when measured); gpt-4o does no reasoning and needs none.
    reasoning_token_overhead: int
    # Whether a provider error may be answered by another provider's model.
    cross_provider_fallback: bool
    # Whether the Jev completeness check and its repair rerun follow pass 4.
    jev_check: bool


# Measured on 2026-10-08 (v2 passes, not v3) over ten conversations
# against gpt-4o on the same prompts, gpt-oss-120b kept 96.5% of load-bearing
# strings (IDs, names, amounts, dates) where gpt-4o kept 91%, at about a
# quarter of the cost. A Groq failure may fall back to another provider, as it
# always has.
GROQ_PROFILE = CompactionProfile(
    name="groq",
    provider=LLMProvider.GROQ,
    model="openai/gpt-oss-120b",
    reasoning_token_overhead=2000,
    cross_provider_fallback=True,
    jev_check=False,
)

# gpt-4o writes, then Jev checks the result for dropped details and pass 2
# repairs what it finds (completeness_check.py). On the same measurement this
# pairing matched Groq's retention (96.5%); gpt-4o without the check did not.
# It stays on OpenAI when a call errors: the token budget here carries no
# reasoning overhead, so handing the call to a Groq reasoning model would
# truncate it.
OPENAI_JEV_PROFILE = CompactionProfile(
    name="openai_jev",
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    reasoning_token_overhead=0,
    cross_provider_fallback=False,
    jev_check=True,
)

# gpt-4o as the writer, with no Jev check. This is the pairing compaction v3 was
# measured with (harnesses/compaction_model_swap, arm openai:gpt-4o), and it is
# the default when this process has an OpenAI key (21 Oct 2026, decided by the
# product owner): the v3 results are for gpt-4o, and no v3 run on Groq's
# gpt-oss-120b exists. SYNAP_COMPACTION_PROFILE=groq or openai_jev still pick
# the others. A failed call stays on OpenAI: the token budget here carries no
# reasoning overhead, and another provider's model was not measured.
OPENAI_PROFILE = CompactionProfile(
    name="openai",
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    reasoning_token_overhead=0,
    cross_provider_fallback=False,
    jev_check=False,
)

COMPACTION_PROFILES = {p.name: p for p in (GROQ_PROFILE, OPENAI_JEV_PROFILE, OPENAI_PROFILE)}
COMPACTION_PROFILE_ENV = "SYNAP_COMPACTION_PROFILE"

# The edition value that selects the profile built from the developer's own
# LLM settings (Core). Cloud's value is None: the three fixed
# profiles above choose as before.
CONFIGURED_PROFILE_NAME = "configured"


class CompactionProfileError(RuntimeError):
    """Compaction was asked to use the configured LLM and none is configured."""


def configured_profile() -> CompactionProfile:
    """The profile for the LLM the developer configured (``SYNAP_LLM_PROVIDER`` and ``SYNAP_LLM_MODEL``).

    ``SYNAP_LLM_MODEL_COMPACTION`` picks a different model for this step only. The
    provider is the configured one and a failed call never moves to another provider.
    No reasoning-token allowance is added, and there is no Jev check.

    Raises ``CompactionProfileError`` when no LLM is configured: a Core compaction
    that cannot reach a model must fail where someone sees it, not fall to a
    provider the developer did not choose.
    """
    settings = llm_settings.active()
    if settings is None:
        raise CompactionProfileError(
            f"Compaction runs on the configured LLM, but {llm_settings.PROVIDER_VAR} and "
            f"{llm_settings.MODEL_VAR} are not set. Set both."
        )
    return CompactionProfile(
        name=CONFIGURED_PROFILE_NAME,
        provider=settings.provider,
        model=settings.model_for("COMPACTION"),
        reasoning_token_overhead=0,
        cross_provider_fallback=False,
        jev_check=False,
    )


def _groq_available(llm_manager: LLMConfigManager) -> bool:
    """Whether this process has a Groq client, i.e. a Groq key was present."""
    try:
        llm_manager.get_provider(LLMProvider.GROQ)
        return True
    except Exception:
        return False


def _openai_available(llm_manager: LLMConfigManager) -> bool:
    """Whether this process has an OpenAI client, i.e. an OpenAI key was present."""
    try:
        llm_manager.get_provider(LLMProvider.OPENAI)
        return True
    except Exception:
        return False


def resolve_compaction_profile(llm_manager: LLMConfigManager) -> CompactionProfile:
    """Pick the compaction profile.

    Where the edition names the configured profile (Core), that profile is used and
    SYNAP_COMPACTION_PROFILE is ignored (logged when set).

    Otherwise SYNAP_COMPACTION_PROFILE, when it names a profile, wins. Otherwise gpt-4o
    (OPENAI_PROFILE) when this process has an OpenAI key, which is the model v3
    was measured with. With no OpenAI key: Groq when it has a Groq key, and
    openai_jev last, so a deployment with neither key still gets a profile
    rather than an error at start.

    An unrecognised value is ignored rather than fatal. Compaction runs in the
    background, and a typo in an env var should not be what stops every
    conversation from being summarised.
    """
    raw = os.environ.get(COMPACTION_PROFILE_ENV, "").strip().lower()
    if edition.compaction_profile == CONFIGURED_PROFILE_NAME:
        if raw:
            logger.warning(
                "%s=%r is ignored: compaction uses the configured LLM", COMPACTION_PROFILE_ENV, raw,
            )
        return configured_profile()
    if raw in COMPACTION_PROFILES:
        return COMPACTION_PROFILES[raw]
    if raw:
        logger.warning(
            "%s=%r is not one of %s; choosing automatically",
            COMPACTION_PROFILE_ENV, raw, sorted(COMPACTION_PROFILES),
        )
    if _openai_available(llm_manager):
        return OPENAI_PROFILE
    return GROQ_PROFILE if _groq_available(llm_manager) else OPENAI_JEV_PROFILE


class BaseCompactionPass(ICompactionPass, ABC):
    """
    Base class for all compaction passes.
    Provides common functionality for LLM calls, timing, and error handling.

    Uses LLMConfigManager with the provider and model of one CompactionProfile.
    """

    # Class-level defaults, kept for callers that read them off the class. A
    # pass instance carries its own profile's values under the same names.
    COMPACTION_PROVIDER = GROQ_PROFILE.provider
    COMPACTION_MODEL = GROQ_PROFILE.model
    REASONING_TOKEN_OVERHEAD = GROQ_PROFILE.reasoning_token_overhead

    # JSON parse retries — OSS models frequently produce malformed JSON
    JSON_MAX_RETRIES = 2

    def __init__(
        self,
        llm_manager: LLMConfigManager,
        profile: Optional[CompactionProfile] = None,
    ):
        self.profile = profile or resolve_compaction_profile(llm_manager)
        self.COMPACTION_PROVIDER = self.profile.provider
        self.COMPACTION_MODEL = self.profile.model
        self.REASONING_TOKEN_OVERHEAD = self.profile.reasoning_token_overhead
        self._llm_manager = llm_manager
        self._tokens_consumed = 0
        self._retry_count = 0

    @property
    @abstractmethod
    def pass_name(self) -> str:
        pass

    @property
    @abstractmethod
    def pass_number(self) -> int:
        pass

    @property
    @abstractmethod
    def temperature(self) -> float:
        """Default temperature for this pass."""
        pass

    async def execute(
        self,
        input_data: Dict[str, Any],
        config: CompactionConfig,
        previous_results: Dict[str, Any]
    ) -> PassResult:
        """
        Execute this pass with timing and error handling.
        """
        start_time = time.time()
        self._tokens_consumed = 0
        self._retry_count = 0

        try:
            output = await self._execute_impl(input_data, config, previous_results)

            latency_ms = (time.time() - start_time) * 1000

            return PassResult(
                pass_name=self.pass_name,
                pass_number=self.pass_number,
                success=True,
                output=output,
                latency_ms=latency_ms,
                tokens_consumed=self._tokens_consumed,
                retry_count=self._retry_count
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            return PassResult(
                pass_name=self.pass_name,
                pass_number=self.pass_number,
                success=False,
                output={},
                latency_ms=latency_ms,
                tokens_consumed=self._tokens_consumed,
                error=str(e),
                retry_count=self._retry_count
            )

    @abstractmethod
    async def _execute_impl(
        self,
        input_data: Dict[str, Any],
        config: CompactionConfig,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Implementation of the pass logic.
        Subclasses must implement this.
        """
        pass

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
        json_mode: bool = False
    ) -> str:
        """
        Make an LLM call through the config manager.
        
        Forces the provider and model of this pass's CompactionProfile.

        Args:
            system_prompt: System instructions
            user_prompt: User prompt
            max_tokens: Maximum response tokens
            json_mode: Whether to request JSON output
            
        Returns:
            LLM response content
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Add reasoning overhead so the requested output capacity is available
        effective_max_tokens = max_tokens + self.REASONING_TOKEN_OVERHEAD

        extra: Dict[str, Any] = {}
        if not self.profile.cross_provider_fallback:
            extra["allow_cross_provider_fallback"] = False

        response = await self._llm_manager.complete(
            messages=messages,
            force_provider=self.COMPACTION_PROVIDER,
            force_model=self.COMPACTION_MODEL,
            temperature=self.temperature,
            max_tokens=effective_max_tokens,
            json_mode=json_mode,
            reasoning_effort="medium",
            **extra,
        )

        self._tokens_consumed += response.total_tokens
        return response.content

    async def _call_llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
    ) -> Dict[str, Any]:
        """Call LLM and parse JSON response with retry on malformed JSON.

        OSS models frequently produce malformed JSON on the first attempt.
        Retries with a stronger JSON instruction on failure.
        """
        last_error = None
        current_system = system_prompt
        for attempt in range(self.JSON_MAX_RETRIES + 1):
            try:
                content = await self._call_llm(
                    system_prompt=current_system,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    json_mode=True,
                )
                return self._parse_json_response(content)
            except json.JSONDecodeError as e:
                last_error = e
                self._retry_count += 1
                if attempt < self.JSON_MAX_RETRIES:
                    logger.warning(
                        "%s: JSON parse failed (attempt %d/%d), retrying: %s",
                        self.pass_name, attempt + 1, self.JSON_MAX_RETRIES + 1, e,
                    )
                    # Imported here: passes/prompts does not ship to Core, and the
                    # retry (the v2 passes) is the only reader.
                    from .prompts.json_retry_prompts import JSON_RETRY_INSTRUCTION
                    current_system = system_prompt + JSON_RETRY_INSTRUCTION
        raise last_error

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Strip markdown code blocks if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        return json.loads(content.strip())

    def _format_messages_for_prompt(
        self,
        messages: List[ConversationMessage],
        include_ids: bool = True
    ) -> str:
        """Format messages for inclusion in prompts."""
        formatted = []
        for msg in messages:
            prefix = f"[{msg.message_id}] " if include_ids else ""
            formatted.append(f"{prefix}{msg.role.upper()}: {msg.content}")
        return "\n\n".join(formatted)

    def _segment_messages(
        self,
        messages: List[ConversationMessage],
        segment_size: int = 10
    ) -> List[List[ConversationMessage]]:
        """Split messages into segments for processing."""
        segments = []
        for i in range(0, len(messages), segment_size):
            segments.append(messages[i:i + segment_size])
        return segments
