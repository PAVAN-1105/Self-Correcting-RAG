"""Document ingestion and semantic chunking for SCRAG."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a chunked segment of an ingested document."""

    chunk_id: str = Field(description="Unique identifier for chunk")
    content: str = Field(description="Textual content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata tags")


class DocumentLoader:
    """Loads and chunks documents from files or directories."""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def load_text(self, text: str, source: str = "raw_text") -> List[DocumentChunk]:
        """Split a raw text string into DocumentChunks."""
        splits = self.splitter.split_text(text)
        chunks = []
        for idx, split in enumerate(splits):
            chunk_id = f"{Path(source).stem}_{idx}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=split.strip(),
                    metadata={
                        "source": source,
                        "chunk_index": idx,
                        "char_count": len(split.strip()),
                    },
                )
            )
        return chunks

    def load_file(self, file_path: str | Path) -> List[DocumentChunk]:
        """Load a file from disk and return chunked documents."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix in [".txt", ".md", ".json", ".csv"]:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return self.load_text(content, source=str(path))
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return self.load_text(content, source=str(path))

    def load_directory(
        self, directory_path: str | Path, recursive: bool = True
    ) -> List[DocumentChunk]:
        """Load all compatible files within a directory."""
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        chunks: List[DocumentChunk] = []
        pattern = "**/*" if recursive else "*"
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in [
                ".txt",
                ".md",
                ".json",
            ]:
                chunks.extend(self.load_file(file_path))
        return chunks
