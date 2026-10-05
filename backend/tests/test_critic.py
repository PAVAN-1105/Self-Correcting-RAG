"""Unit tests for Grounded Generation, Faithfulness Critic, and Relevance Critic in SCRAG."""

import pytest

from scrag.config import SCRAGSettings
from scrag.critic.faithfulness import FaithfulnessCritic
from scrag.critic.relevance import AnswerRelevanceCritic
from scrag.generator.generator import GroundedGenerator
from scrag.state import CritiqueType


@pytest.fixture
def critic_setup():
    settings = SCRAGSettings(faithfulness_threshold=0.60, relevance_threshold=0.60)
    generator = GroundedGenerator(settings=settings, use_llm=False)
    faithfulness = FaithfulnessCritic(settings=settings, use_llm=False)
    relevance = AnswerRelevanceCritic(settings=settings, use_llm=False)
    return {
        "settings": settings,
        "generator": generator,
        "faithfulness": faithfulness,
        "relevance": relevance,
    }


def test_grounded_generator(critic_setup):
    generator = critic_setup["generator"]
    question = "What is SCRAG?"
    context = "[1] (Document: scrag.txt)\nSCRAG is Self Correcting Retrieval Augmented Generation."

    result = generator.generate(question, context)
    assert len(result.text) > 0
    assert len(result.citations) > 0
    assert "[1]" in result.citations


def test_faithfulness_critic_pass(critic_setup):
    faithfulness = critic_setup["faithfulness"]
    context = "[1] (Doc: sample.txt)\nParis is the capital of France and has the Eiffel Tower."
    draft = "According to verified sources [1], Paris is the capital of France."

    critique = faithfulness.evaluate(draft, context)
    assert critique.critique_type == CritiqueType.FAITHFULNESS
    assert critique.passed is True
    assert critique.score >= 0.60


def test_faithfulness_critic_hallucination_detected(critic_setup):
    faithfulness = critic_setup["faithfulness"]
    context = "[1] (Doc: sample.txt)\nParis is the capital of France."
    draft = "Paris is the capital of France and was founded in 5000 BC by fictional aliens (hallucinated content)."

    critique = faithfulness.evaluate(draft, context)
    assert critique.critique_type == CritiqueType.FAITHFULNESS
    assert critique.passed is False
    assert len(critique.hallucinated_claims) > 0


def test_relevance_critic_pass(critic_setup):
    relevance = critic_setup["relevance"]
    question = "Who wrote Hamlet?"
    draft = "Hamlet was written by William Shakespeare."

    critique = relevance.evaluate(question, draft)
    assert critique.critique_type == CritiqueType.ANSWER_RELEVANCE
    assert critique.passed is True
    assert critique.score >= 0.60


def test_relevance_critic_fail_empty(critic_setup):
    relevance = critic_setup["relevance"]
    question = "What is machine learning?"
    draft = ""

    critique = relevance.evaluate(question, draft)
    assert critique.passed is False
    assert critique.score == 0.0
