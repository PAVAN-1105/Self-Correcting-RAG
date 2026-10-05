"""SCRAG Document Grader: assesses retrieval relevance and triggers corrective actions."""

import re
from typing import List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from scrag.config import SCRAGSettings, get_settings
from scrag.retrieval.loaders import DocumentChunk
from scrag.state import RetrievalAction, RetrievalGrade


class GradeOutput(BaseModel):
    """Structured evaluation output for document relevance."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance confidence score between 0.0 (irrelevant) and 1.0 (highly relevant)",
    )
    is_relevant: bool = Field(
        description="True if the document contains useful facts to answer the question, False otherwise"
    )
    reasoning: str = Field(description="Concise justification for the grading score")


class DocumentGrader:
    """Evaluates whether retrieved documents are relevant to the user query."""

    def __init__(
        self,
        settings: Optional[SCRAGSettings] = None,
        llm: Optional[BaseChatModel] = None,
        use_llm: bool = True,
    ):
        self.settings = settings or get_settings()

        if not use_llm:
            self.llm = None
        elif llm is not None:
            self.llm = llm
        else:
            from scrag.llm_factory import get_critic_llm

            raw_llm = get_critic_llm(self.settings)
            if raw_llm is not None:
                try:
                    self.llm = raw_llm.with_structured_output(GradeOutput)
                except Exception:
                    self.llm = raw_llm
            else:
                self.llm = None

    def grade_document(
        self, question: str, chunk: DocumentChunk
    ) -> RetrievalGrade:
        """Grade a single document chunk for relevance to the question."""
        if self.llm is not None:
            system_prompt = (
                "You are an expert retrieval evaluator assessing document relevance. "
                "Given a user query and a retrieved document snippet, evaluate whether "
                "the document contains information that can help answer the question. "
                "Be objective and strict: tangential or off-topic information is not relevant."
            )
            user_prompt = f"Question: {question}\n\nDocument Snippet:\n{chunk.content}"

            try:
                res = self.llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )
                if isinstance(res, dict):
                    res = GradeOutput(**res)
                elif hasattr(res, "content") and not isinstance(res, GradeOutput):
                    import json
                    match = re.search(r"\{.*\}", str(res.content), re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
                        res = GradeOutput(**data)

                if isinstance(res, GradeOutput):
                    return RetrievalGrade(
                        document_id=chunk.chunk_id,
                        source=chunk.metadata.get("source", "internal"),
                        score=res.score,
                        is_relevant=res.is_relevant,
                        reasoning=res.reasoning,
                    )
            except Exception:
                pass

        return self._heuristic_grade(question, chunk)

    def grade_all(
        self, question: str, chunks: List[DocumentChunk]
    ) -> Tuple[List[RetrievalGrade], RetrievalAction]:
        """Grade all retrieved chunks and determine the SCRAG action pathway."""
        if not chunks:
            return [], RetrievalAction.INCORRECT

        grades: List[RetrievalGrade] = []
        for chunk in chunks:
            grade = self.grade_document(question, chunk)
            grades.append(grade)

        scores = [g.score for g in grades]
        max_score = max(scores) if scores else 0.0

        if max_score >= self.settings.crag_upper_threshold:
            action = RetrievalAction.CORRECT
        elif max_score < self.settings.crag_lower_threshold:
            action = RetrievalAction.INCORRECT
        else:
            action = RetrievalAction.AMBIGUOUS

        return grades, action

    def _heuristic_grade(
        self, question: str, chunk: DocumentChunk
    ) -> RetrievalGrade:
        """Deterministic lexical overlap score for testing and offline fallbacks."""
        stopwords = {
            "what", "is", "the", "and", "how", "does", "are", "for", "with",
            "that", "this", "can", "you", "tell", "about", "from", "into"
        }
        all_q_words = re.findall(r"\b\w{3,}\b", question.lower())
        q_words = set(w for w in all_q_words if w not in stopwords)
        doc_words = set(re.findall(r"\b\w{3,}\b", chunk.content.lower()))

        if not q_words:
            q_words = set(all_q_words)

        if not q_words:
            return RetrievalGrade(
                document_id=chunk.chunk_id,
                source=chunk.metadata.get("source", "internal"),
                score=0.1,
                is_relevant=False,
                reasoning="Empty query.",
            )

        overlap = len(q_words.intersection(doc_words))
        score = overlap / len(q_words) if q_words else 0.0
        score = min(1.0, max(0.0, score))
        is_rel = score >= 0.25 or overlap >= 1

        return RetrievalGrade(
            document_id=chunk.chunk_id,
            source=chunk.metadata.get("source", "internal"),
            score=round(score, 2),
            is_relevant=is_rel,
            reasoning=f"Lexical overlap: {overlap} matching query terms ({score:.2f})",
        )
