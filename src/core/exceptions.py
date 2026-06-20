class MiniCodeError(Exception):
    """Base exception for MiniCode system."""


class ConfigurationError(MiniCodeError):
    """Invalid configuration."""


class LLMServiceError(MiniCodeError):
    """DeepSeek API call failed."""


class StorageError(MiniCodeError):
    """Database or cache operation failed."""


class SkillRoutingError(MiniCodeError):
    """Skill selection or routing failed."""


class ContextCompressionError(MiniCodeError):
    """Context compression or decompression failed."""


class BugLocalizationError(MiniCodeError):
    """Bug localization failed."""


class PatchGenerationError(MiniCodeError):
    """Patch generation failed."""


class VerificationError(MiniCodeError):
    """Test verification failed."""


class HumanInLoopInterruptError(MiniCodeError):
    """Waiting for human review."""


class EmbeddingError(MiniCodeError):
    """Embedding generation failed."""
