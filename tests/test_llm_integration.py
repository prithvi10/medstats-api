"""
Tests for LLM service integration.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.fixture
def mock_settings():
    """Mock settings with an API key."""
    with patch("src.agentic_core.llm.service.settings") as mock:
        mock.OPENAI_API_KEY = "sk-test-key-12345"
        mock.OPENAI_MODEL = "gpt-5"
        mock.OPENAI_MODEL_MINI = "gpt-4o-mini"
        mock.OPENAI_MODEL_REASONING = "gpt-5"
        yield mock


@pytest.fixture
def mock_openai_client():
    """Mock AsyncOpenAI client."""
    with patch("src.agentic_core.llm.service.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


class TestLLMServiceInit:
    def test_raises_without_api_key(self):
        """LLMService raises ValueError when no API key is configured."""
        with patch("src.agentic_core.llm.service.settings") as mock:
            mock.OPENAI_API_KEY = ""
            from src.agentic_core.llm.service import LLMService
            with pytest.raises(ValueError, match="OPENAI_API_KEY not set"):
                LLMService()

    def test_creates_with_api_key(self, mock_settings, mock_openai_client):
        """LLMService initializes correctly with API key."""
        from src.agentic_core.llm.service import LLMService
        service = LLMService()
        assert service.default_model == "gpt-5"
        assert service.call_count == 0

    def test_get_llm_service_returns_none_without_key(self):
        """get_llm_service returns None when no key is set."""
        with patch("src.agentic_core.llm.service.settings") as mock:
            mock.OPENAI_API_KEY = ""
            from src.agentic_core.llm.service import get_llm_service
            assert get_llm_service() is None


class TestLLMServiceGenerateText:
    @pytest.mark.asyncio
    async def test_generate_text_calls_openai(self, mock_settings, mock_openai_client):
        """generate_text calls OpenAI chat completions with correct params."""
        from src.agentic_core.llm.service import LLMService

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated narrative text"
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=30)

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        service = LLMService()
        result = await service.generate_text(
            prompt="Describe this variable",
            system_prompt="You are a biostatistician",
            temperature=0.3,
        )

        assert result == "Generated narrative text"
        assert service.total_prompt_tokens == 50
        assert service.total_completion_tokens == 30
        assert service.call_count == 1

        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-5"
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"


class TestLLMServiceGenerateJSON:
    @pytest.mark.asyncio
    async def test_generate_json_returns_dict(self, mock_settings, mock_openai_client):
        """generate_json parses valid JSON response into dict."""
        from src.agentic_core.llm.service import LLMService

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"issues": [], "overall": "ok"}'
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=20)

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        service = LLMService()
        result = await service.generate_json(prompt="Review this report")

        assert result == {"issues": [], "overall": "ok"}
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_generate_json_handles_invalid_json(self, mock_settings, mock_openai_client):
        """generate_json returns empty dict on invalid JSON."""
        from src.agentic_core.llm.service import LLMService

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Not valid JSON at all"
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=10)

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        service = LLMService()
        result = await service.generate_json(prompt="Some prompt")

        assert result == {}


class TestLLMServiceStructured:
    @pytest.mark.asyncio
    async def test_generate_structured_parses_response(self, mock_settings, mock_openai_client):
        """generate_structured uses beta.chat.completions.parse."""
        from pydantic import BaseModel
        from src.agentic_core.llm.service import LLMService

        class TestModel(BaseModel):
            variable: str
            appropriate: bool

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = TestModel(variable="AGE", appropriate=True)
        mock_response.usage = MagicMock(prompt_tokens=80, completion_tokens=15)

        mock_openai_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

        service = LLMService()
        result = await service.generate_structured(
            prompt="Check this variable",
            response_format=TestModel,
        )

        assert result.variable == "AGE"
        assert result.appropriate is True


class TestLLMServiceUsageTracking:
    @pytest.mark.asyncio
    async def test_cumulative_usage_tracking(self, mock_settings, mock_openai_client):
        """Usage tracking accumulates across multiple calls."""
        from src.agentic_core.llm.service import LLMService

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "text"
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        service = LLMService()
        await service.generate_text("call 1")
        await service.generate_text("call 2")

        summary = service.get_usage_summary()
        assert summary["total_prompt_tokens"] == 200
        assert summary["total_completion_tokens"] == 100
        assert summary["total_tokens"] == 300
        assert summary["call_count"] == 2
        assert summary["model"] == "gpt-5"

    @pytest.mark.asyncio
    async def test_reset_usage(self, mock_settings, mock_openai_client):
        """reset_usage clears all counters."""
        from src.agentic_core.llm.service import LLMService

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "text"
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        service = LLMService()
        await service.generate_text("call 1")
        service.reset_usage()

        assert service.total_prompt_tokens == 0
        assert service.total_completion_tokens == 0
        assert service.call_count == 0


class TestAgentLLMIntegration:
    """Test that agents correctly use LLM service when provided."""

    @pytest.mark.asyncio
    async def test_report_generator_uses_llm_for_narrative(self, mock_settings, mock_openai_client):
        """ReportGeneratorAgent falls through to LLM for narrative when service provided."""
        from src.agentic_core.llm.service import LLMService
        from src.medstats.agents.report_generator import ReportGeneratorAgent
        from src.agentic_core.agents.state import MissionState

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "AI-generated narrative about age distribution."
        mock_response.usage = MagicMock(prompt_tokens=200, completion_tokens=80)

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        llm = LLMService()
        agent = ReportGeneratorAgent(services=llm)

        state = MissionState(submission_id="test-1", job_id="job-1")
        state.agent_outputs["title"] = "Test Study"
        state.agent_outputs["sample_size"] = 100
        state.agent_outputs["study_design"] = "single_cohort"
        state.agent_outputs["variables"] = [{"name": "AGE", "dtype": "numerical"}]
        state.agent_outputs["test_results"] = [{
            "variable_name": "AGE",
            "descriptive": {"mean": 45.2, "std": 12.3, "min": 18, "max": 78},
            "frequency_table": {
                "rows": [
                    {"category": "18-30", "count": 20, "percentage": 20.0},
                    {"category": "31-50", "count": 50, "percentage": 50.0},
                    {"category": "51-78", "count": 30, "percentage": 30.0},
                ]
            },
            "test_result": {"test_type": "shapiro_wilk", "p_value": 0.12, "is_significant_05": False,
                            "significance_statement": "Data is normally distributed (p=0.12)"},
        }]

        result_state = await agent.run(state)

        assert result_state.status != "FAILED"
        sections = result_state.agent_outputs.get("report_sections", [])
        narratives = [s for s in sections if s.get("type") == "paragraph"]
        assert any("AI-generated" in n.get("content", "") for n in narratives)

    @pytest.mark.asyncio
    async def test_report_generator_fallback_without_llm(self):
        """ReportGeneratorAgent uses template fallback when no LLM service."""
        from src.medstats.agents.report_generator import ReportGeneratorAgent
        from src.agentic_core.agents.state import MissionState

        agent = ReportGeneratorAgent(services=None)

        state = MissionState(submission_id="test-2", job_id="job-2")
        state.agent_outputs["title"] = "Fallback Study"
        state.agent_outputs["sample_size"] = 50
        state.agent_outputs["study_design"] = "single_cohort"
        state.agent_outputs["variables"] = [{"name": "SEX", "dtype": "categorical"}]
        state.agent_outputs["test_results"] = [{
            "variable_name": "SEX",
            "descriptive": {"mean": None, "std": None, "categories": ["Male", "Female"]},
            "frequency_table": {
                "rows": [
                    {"category": "Male", "count": 30, "percentage": 60.0},
                    {"category": "Female", "count": 20, "percentage": 40.0},
                    {"category": "TOTAL", "count": 50, "percentage": 100.0},
                ]
            },
            "test_result": None,
        }]

        result_state = await agent.run(state)
        assert result_state.status != "FAILED"
        sections = result_state.agent_outputs.get("report_sections", [])
        assert len(sections) > 0
