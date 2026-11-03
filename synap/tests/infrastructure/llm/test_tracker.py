"""
Tests for LLM Usage Tracker.
"""

import pytest
from synap.cloud.infrastructure.llm import LLMProvider


class TestLLMUsageTracker:
    """Test usage tracking and availability calculation."""

    @pytest.mark.asyncio
    async def test_record_request(self, tracker):
        """Test recording a request."""
        await tracker.record_request(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50
        )

        # Verify counters were incremented
        availability = await tracker.get_availability(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            rpm_limit=1000,
            tpm_limit=100000
        )
        assert availability.rpm_usage > 0
        assert availability.tpm_usage > 0

    @pytest.mark.asyncio
    async def test_record_error(self, tracker):
        """Test recording an error."""
        await tracker.record_error(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            is_rate_limit=False
        )

        # Should not immediately block availability (need 3 errors)
        availability = await tracker.get_availability(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            rpm_limit=1000,
            tpm_limit=100000
        )
        assert availability.is_available

    @pytest.mark.asyncio
    async def test_availability_calculation_light_load(self, tracker):
        """Test availability calculation with light load."""
        # Record minimal usage
        await tracker.record_request(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            input_tokens=100,
            output_tokens=100
        )

        availability = await tracker.get_availability(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            rpm_limit=1000,
            tpm_limit=100000
        )
        
        assert availability.is_available
        assert availability.availability_factor == 1.0
        assert availability.rpm_usage < 0.5

    @pytest.mark.asyncio
    async def test_availability_calculation_heavy_load(self, tracker):
        """Test availability calculation with heavy load."""
        # Record many requests to simulate heavy load
        for _ in range(85):
            await tracker.record_request(
                provider=LLMProvider.OPENAI,
                model="gpt-4o",
                input_tokens=1000,
                output_tokens=1000
            )

        availability = await tracker.get_availability(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            rpm_limit=100,
            tpm_limit=200000
        )
        
        assert availability.is_available
        assert availability.availability_factor == 0.5  # Heavy load
        assert availability.rpm_usage > 0.8
        assert availability.reason == "Heavy load"

    @pytest.mark.asyncio
    async def test_availability_exhausted(self, tracker):
        """Test availability when rate limit exhausted."""
        # Exhaust rate limit
        for _ in range(101):
            await tracker.record_request(
                provider=LLMProvider.OPENAI,
                model="gpt-4o",
                input_tokens=100,
                output_tokens=100
            )

        availability = await tracker.get_availability(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            rpm_limit=100,
            tpm_limit=100000
        )
        
        assert not availability.is_available
        assert availability.availability_factor == 0.0
        assert availability.reason == "Rate limit exhausted"

    @pytest.mark.asyncio
    async def test_mark_unavailable(self, tracker):
        """Test marking model as unavailable."""
        await tracker.mark_unavailable(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            duration_seconds=60
        )

        availability = await tracker.get_availability(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            rpm_limit=1000,
            tpm_limit=100000
        )
        
        assert not availability.is_available
        assert availability.reason == "Too many recent errors"

    @pytest.mark.asyncio
    async def test_clear_usage(self, tracker):
        """Test clearing usage data."""
        # Record some usage
        await tracker.record_request(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            input_tokens=100,
            output_tokens=100
        )

        # Clear it
        await tracker.clear_usage(
            provider=LLMProvider.OPENAI,
            model="gpt-4o"
        )

        # Should be back to zero
        availability = await tracker.get_availability(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            rpm_limit=1000,
            tpm_limit=100000
        )
        assert availability.rpm_usage == 0
        assert availability.tpm_usage == 0

    @pytest.mark.asyncio
    async def test_multiple_models_isolated(self, tracker):
        """Test that usage tracking is isolated per model."""
        # Record usage for model 1
        await tracker.record_request(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=1000
        )

        # Check model 2 is unaffected
        availability = await tracker.get_availability(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            rpm_limit=1000,
            tpm_limit=100000
        )
        assert availability.rpm_usage == 0
        assert availability.tpm_usage == 0
