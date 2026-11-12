"""Custom application exceptions with clear messages."""


class AskDocsError(Exception):
    """Base exception for AskDocs."""


class DocumentNotFoundError(AskDocsError):
    def __init__(self, doc_id: str) -> None:
        super().__init__(f"Document '{doc_id}' not found.")
        self.doc_id = doc_id


class UnsupportedFileTypeError(AskDocsError):
    def __init__(self, filename: str, allowed: list[str]) -> None:
        super().__init__(
            f"File '{filename}' has an unsupported type. Allowed: {allowed}"
        )


class FileTooLargeError(AskDocsError):
    def __init__(self, filename: str, max_mb: int) -> None:
        super().__init__(f"File '{filename}' exceeds the {max_mb} MB limit.")


class VectorStoreError(AskDocsError):
    """Raised when FAISS operations fail."""


class EmbeddingError(AskDocsError):
    """Raised when an embedding API call fails."""


class LLMError(AskDocsError):
    """Raised when an LLM API call fails."""


class CacheError(AskDocsError):
    """Raised when a cache operation fails (non-fatal, use fallback)."""
