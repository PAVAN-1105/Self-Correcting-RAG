"""SCRAG Knowledge Refinement: Decompose-then-recompose strip extraction algorithm."""

import re
from typing import List, Optional

from scrag.config import SCRAGSettings, get_settings
from scrag.evaluator.grader import DocumentGrader
from scrag.retrieval.loaders import DocumentChunk
from scrag.state import KnowledgeStrip


class KnowledgeRefiner:
    """Implements CRAG's Decompose-then-Recompose algorithm."""

    def __init__(
        self,
        settings: Optional[SCRAGSettings] = None,
        grader: Optional[DocumentGrader] = None,
    ):
        self.settings = settings or get_settings()
        self.grader = grader or DocumentGrader(settings=self.settings)

    def decompose(
        self, chunks: List[DocumentChunk], is_external: bool = False
    ) -> List[KnowledgeStrip]:
        """Decompose chunks into fine-grained atomic knowledge strips."""
        strips: List[KnowledgeStrip] = []

        for chunk in chunks:
            raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", chunk.content)
            sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15]

            for i in range(0, len(sentences), 2):
                strip_text = " ".join(sentences[i : i + 2]).strip()
                if not strip_text:
                    continue

                strip_id = f"{chunk.chunk_id}_s{i // 2}"
                source = chunk.metadata.get("source", "internal")

                strips.append(
                    KnowledgeStrip(
                        strip_id=strip_id,
                        source=source,
                        content=strip_text,
                        score=1.0,
                        is_relevant=True,
                        is_external=is_external,
                    )
                )

        return strips

    def filter_and_recompose(
        self,
        question: str,
        strips: List[KnowledgeStrip],
        top_k: Optional[int] = None,
    ) -> List[KnowledgeStrip]:
        """Filter out noisy strips and recompose top relevant strips."""
        if not strips:
            return []

        limit = top_k or self.settings.crag_strip_top_k
        scored_strips: List[KnowledgeStrip] = []

        for strip in strips:
            temp_chunk = DocumentChunk(
                chunk_id=strip.strip_id,
                content=strip.content,
                metadata={"source": strip.source},
            )
            grade = self.grader.grade_document(question, temp_chunk)

            if grade.is_relevant and grade.score >= 0.25:
                refined_strip = strip.model_copy(deep=True)
                refined_strip.score = grade.score
                refined_strip.is_relevant = True
                scored_strips.append(refined_strip)

        scored_strips.sort(key=lambda s: s.score, reverse=True)
        return scored_strips[:limit]

    def format_for_generation(self, strips: List[KnowledgeStrip]) -> str:
        """Format recomposed strips into an annotated context string for the generator."""
        if not strips:
            return "No verified context available."

        formatted_blocks = []
        for idx, strip in enumerate(strips, start=1):
            source_type = "Web" if strip.is_external else "Document"
            formatted_blocks.append(
                f"[{idx}] ({source_type}: {strip.source})\n{strip.content}"
            )

        return "\n\n".join(formatted_blocks)
