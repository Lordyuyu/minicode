"""
Vector literal utilities for pgvector SQL operations.

Embedding vectors are always ``list[float]`` from embedding models —
every element is a Python float. The ``_vector_literal`` function builds
a pgvector-compatible cast expression (``'[...]'::vector``) for use in
raw SQL ``text()`` statements.

**Security note:** These values originate from deterministic embedding
model outputs and are always numeric. The ``float(v)`` coercion validates
this at runtime; a non-numeric input raises ``TypeError`` or ``ValueError``
before any SQL is constructed. This is safer than blind f-string
interpolation while remaining compatible with SQLAlchemy ``text()``
queries that cannot natively bind pgvector types.
"""

from __future__ import annotations

from collections.abc import Sequence


def vector_literal(vec: Sequence[float]) -> str:
    """Build a pgvector cast literal from a sequence of floats.

    Each element is coerced through ``float()`` to guarantee only numeric
    values reach the SQL string. The resulting literal can be safely
    interpolated into a ``sqlalchemy.text()`` query.

    Args:
        vec: Embedding vector as a sequence of floats.

    Returns:
        A string like ``'[0.12,-0.34,0.56]'::vector`` suitable for
        direct use in a pgvector SQL expression.

    Raises:
        TypeError: If any element cannot be converted to ``float``.
    """
    # float() coercion is the safety boundary — non-numeric inputs raise
    formatted = ",".join(str(float(v)) for v in vec)
    return f"'[{formatted}]'::vector"
