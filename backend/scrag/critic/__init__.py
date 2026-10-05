"""SCRAG Post-generation Reflection Critic: Faithfulness (Grounding) and Answer Relevance."""

from scrag.critic.faithfulness import FaithfulnessCritic
from scrag.critic.relevance import AnswerRelevanceCritic

__all__ = ["FaithfulnessCritic", "AnswerRelevanceCritic"]
