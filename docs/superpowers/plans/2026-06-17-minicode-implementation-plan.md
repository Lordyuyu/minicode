# MiniCode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build MiniCode, an end-to-end AI Coding Agent system for automated code repair, featuring a hybrid architecture of LangGraph orchestration + self-developed core algorithms. Aligns with Claude Code-class architecture covering multi-agent collaboration, skill routing, self-evolving memory, hierarchical context compression, prompt cache, and layered security.

**Architecture:** LangGraph StateGraph drives a centralized multi-agent collaboration architecture: a main Orchestrator Agent handles intent recognition, planning, and quality control, while sub-agents execute with permission-scoped Tool Calls. PostgreSQL + pgvector unifies business metadata and vector embeddings with transactional consistency. Redis powers high-speed caching, hierarchical context compression storage, and prompt cache. Seven self-developed core algorithms — Centralized Multi-Agent Orchestration, Dynamic Skill Registration & Routing, Prompt Cache + Hierarchical Context Cache, Dual-layer Skill Routing, Hierarchical Context Compression, Self-evolving Memory, Multi-layer Safety Review — are implemented from scratch atop this infrastructure, delivering an end-to-end pipeline that takes a Python codebase with failing unit tests and produces localized bug reports, generated patches, verified test results, and auto-registered skill assets.

**Tech Stack:** Python 3.12, LangGraph 0.3+, PostgreSQL + pgvector, Redis, DeepSeek API (OpenAI-compatible), Pydantic v2, pytest, Docker, SQLAlchemy 2.0 Async, httpx, loguru, numpy, hashlib

**Plan Location:** `docs/superpowers/plans/`

---

## File Structure

```
E:\project\Mini code\
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── src\
│   ├── __init__.py
│   ├── main.py
│   ├── config\
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── core\
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── state.py
│   │   └── exceptions.py
│   ├── orchestration\
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── react_loop.py
│   │   └── nodes\
│   │       ├── __init__.py
│   │       ├── intent.py
│   │       ├── plan.py
│   │       ├── execute.py
│   │       └── verify.py
│   ├── storage\
│   │   ├── __init__.py
│   │   ├── postgres.py
│   │   ├── redis_client.py
│   │   ├── repository\
│   │   │   ├── __init__.py
│   │   │   ├── task_repo.py
│   │   │   ├── skill_repo.py
│   │   │   └── memory_repo.py
│   │   └── migrations\
│   │       └── 001_init.sql
│   ├── engine\
│   │   ├── __init__.py
│   │   ├── orchestrator\
│   │   │   ├── __init__.py
│   │   │   ├── coordinator.py
│   │   │   └── permission.py
│   │   ├── skill_router\
│   │   │   ├── __init__.py
│   │   │   ├── vector_ranker.py
│   │   │   ├── llm_ranker.py
│   │   │   ├── router.py
│   │   │   └── registry.py
│   │   ├── prompt_cache\
│   │   │   ├── __init__.py
│   │   │   ├── cache_manager.py
│   │   │   └── content_addressed.py
│   │   ├── context_compressor\
│   │   │   ├── __init__.py
│   │   │   ├── hierarchical.py
│   │   │   ├── summarizer.py
│   │   │   └── decompressor.py
│   │   ├── memory\
│   │   │   ├── __init__.py
│   │   │   ├── episodic.py
│   │   │   ├── procedural.py
│   │   │   └── consolidator.py
│   │   └── safety\
│   │       ├── __init__.py
│   │       ├── human_in_loop.py
│   │       └── validator.py
│   ├── llm\
│   │   ├── __init__.py
│   │   ├── deepseek_client.py
│   │   ├── prompt_templates.py
│   │   └── tool_parser.py
│   ├── agents\
│   │   ├── __init__.py
│   │   ├── bug_locator.py
│   │   ├── patch_generator.py
│   │   └── test_verifier.py
│   └── utils\
│       ├── __init__.py
│       ├── embedding.py
│       └── logger.py
├── tests\
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit\
│   │   ├── __init__.py
│   │   ├── test_orchestrator.py
│   │   ├── test_skill_registry.py
│   │   ├── test_prompt_cache.py
│   │   ├── test_safety.py
│   │   ├── test_skill_router.py
│   │   ├── test_context_compressor.py
│   │   ├── test_memory.py
│   │   └── test_deepseek_client.py
│   ├── integration\
│   │   ├── __init__.py
│   │   ├── test_orchestration.py
│   │   └── test_storage.py
│   └── e2e\
│       ├── __init__.py
│       └── test_full_pipeline.py
├── docs\
│   ├── architecture.md
│   └── performance.md
└── samples\
    ├── __init__.py
    └── calculator\
        ├── __init__.py
        ├── calculator.py
        └── test_calculator.py
```

---

### Task 1: Project Scaffolding and Infrastructure

**Files:**
- Create: `pyproject.toml`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/config/__init__.py`
- Create: `src/config/settings.py`
- Create: `src/core/__init__.py`
- Create: `src/core/types.py`
- Create: `src/core/state.py`
- Create: `src/core/exceptions.py`
- Create: `src/utils/__init__.py`
- Create: `src/utils/embedding.py`
- Create: `src/utils/logger.py`
- Create: `src/main.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `samples/__init__.py`

- [ ] **Step 1: Create pyproject.toml with dependencies**

```toml
[project]
name = "minicode"
version = "0.1.0"
description = "AI Coding Agent for automated code repair"
requires-python = ">=3.12"
dependencies = [
    "langgraph>=0.3.0",
    "langchain-core>=0.3.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "asyncpg>=0.30.0",
    "pgvector>=0.3.0",
    "redis[hiredis]>=5.2.0",
    "openai>=1.0.0",
    "httpx>=0.28.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "loguru>=0.7.0",
    "numpy>=2.0.0",
    "alembic>=1.14.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.7.0",
    "mypy>=1.13.0",
    "pre-commit>=4.0.0",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
strict = true
python_version = "3.12"
disallow_untyped_defs = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
version: "3.9"

services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_DB: minicode
      POSTGRES_USER: minicode
      POSTGRES_PASSWORD: minicode_dev
    ports:
      - "15432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U minicode"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "16379:6379"
    volumes:
      - redisdata:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
  redisdata:
```

- [ ] **Step 3: Create .env.example**

```bash
# DeepSeek
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# PostgreSQL
POSTGRES_DSN=postgresql+asyncpg://minicode:minicode_dev@localhost:15432/minicode

# Redis
REDIS_URL=redis://localhost:16379/0

# MiniCode
LOG_LEVEL=DEBUG
MAX_CONTEXT_TOKENS=128000
HUMAN_IN_LOOP_ENABLED=true
SIMILARITY_TOP_K=10
```

- [ ] **Step 4: Create src/config/settings.py**

```python
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    deepseek_api_key: str = Field(alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL")

    postgres_dsn: str = Field(alias="POSTGRES_DSN")
    redis_url: str = Field(default="redis://localhost:16379/0", alias="REDIS_URL")

    log_level: str = Field(default="DEBUG", alias="LOG_LEVEL")
    max_context_tokens: int = Field(default=128000, alias="MAX_CONTEXT_TOKENS")
    human_in_loop_enabled: bool = Field(default=True, alias="HUMAN_IN_LOOP_ENABLED")
    similarity_top_k: int = Field(default=10, alias="SIMILARITY_TOP_K")


settings = Settings()
```

- [ ] **Step 5: Create src/core/types.py with all core type definitions**

```python
from __future__ import annotations

from enum import StrEnum, auto
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


class AgentPhase(StrEnum):
    INTENT_RECOGNITION = auto()
    PLANNING = auto()
    EXECUTION = auto()
    VERIFICATION = auto()


class SkillCategory(StrEnum):
    BUG_LOCALIZATION = auto()
    PATCH_GENERATION = auto()
    TEST_VERIFICATION = auto()
    CODE_ANALYSIS = auto()
    DEPENDENCY_RESOLUTION = auto()


class MemoryType(StrEnum):
    EPISODIC = auto()
    PROCEDURAL = auto()
    SEMANTIC = auto()


@dataclass
class Skill:
    skill_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    category: SkillCategory = SkillCategory.CODE_ANALYSIS
    description: str = ""
    embedding: list[float] = field(default_factory=list)
    success_rate: float = 0.0
    invocation_count: int = 0


@dataclass
class ContextChunk:
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    summary: str = ""
    token_count: int = 0
    compressed: bool = False
    storage_key: str = ""


@dataclass
class MemoryEntry:
    memory_id: str = field(default_factory=lambda: str(uuid4()))
    memory_type: MemoryType = MemoryType.EPISODIC
    task_description: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    outcome: str = ""
    embedding: list[float] = field(default_factory=list)
    procedural_pattern: str = ""
    timestamp: float = 0.0


@dataclass
class PatchResult:
    file_path: str = ""
    original_content: str = ""
    patched_content: str = ""
    diff: str = ""
    verified: bool = False
    verification_output: str = ""


@dataclass
class BugReport:
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    error_type: str = ""
    error_message: str = ""
    confidence: float = 0.0


class PermissionScope(StrEnum):
    READ_ONLY = auto()
    CODE_WRITE = auto()
    TEST_EXEC = auto()
    FULL_ACCESS = auto()


@dataclass
class SubAgentTask:
    agent_id: str = field(default_factory=lambda: str(uuid4()))
    agent_type: str = ""
    task_description: str = ""
    permission: PermissionScope = PermissionScope.READ_ONLY
    target_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    result: str = ""
    success: bool = False


@dataclass
class SkillV2(Skill):
    tags: list[str] = field(default_factory=list)
    applicability: str = ""
    examples: list[str] = field(default_factory=list)
    boundaries: str = ""
    directory: str = ""
```

- [ ] **Step 6: Create src/core/state.py – AgentState dataclass**
- [ ] **Step 7: Create src/core/exceptions.py – custom exceptions**
- [ ] **Step 8: Create src/utils/embedding.py – numpy-based embedding utility**
- [ ] **Step 9: Create src/utils/logger.py – loguru-based logger**
- [ ] **Step 10: Create src/main.py – entry point**
- [ ] **Step 11: Create __init__.py files for all packages**

---

### Task 2: LLM Integration

**Files:**
- Create: `src/llm/__init__.py`
- Create: `src/llm/deepseek_client.py`
- Create: `src/llm/prompt_templates.py`
- Create: `src/llm/tool_parser.py`

- [ ] **Step 1: Implement DeepSeekClient** – OpenAI-compatible wrapper for chat & embedding
- [ ] **Step 2: Create prompt templates** – system prompts for each agent phase
- [ ] **Step 3: Implement tool call parser** – extract structured tool calls from LLM response

---

### Task 3: Storage Layer

**Files:**
- Create: `src/storage/__init__.py`
- Create: `src/storage/postgres.py`
- Create: `src/storage/redis_client.py`
- Create: `src/storage/migrations/001_init.sql`
- Create: `src/storage/repository/__init__.py`
- Create: `src/storage/repository/task_repo.py`
- Create: `src/storage/repository/skill_repo.py`
- Create: `src/storage/repository/memory_repo.py`

- [ ] **Step 1: Implement PostgresClient** – async SQLAlchemy engine + session factory
- [ ] **Step 2: Implement RedisClient** – connection pool + basic ops (set, get, delete, exists)
- [ ] **Step 3: Create SQL migration** – tables for tasks, skills (with pgvector), memory entries, agent logs
- [ ] **Step 4: Implement TaskRepository** – CRUD for tasks
- [ ] **Step 5: Implement SkillRepository** – pgvector ANN search + CRUD
- [ ] **Step 6: Implement MemoryRepository** – store and query memory entries

---

### Task 4: Core Engine – Multi-Agent Orchestration

**Files:**
- Create: `src/engine/__init__.py`
- Create: `src/engine/orchestrator/__init__.py`
- Create: `src/engine/orchestrator/coordinator.py`
- Create: `src/engine/orchestrator/permission.py`

**Integration test:** `tests/integration/test_orchestration.py`

- [ ] **Step 1: Implement PermissionController** – scope-based tool access matrix
- [ ] **Step 2: Implement OrchestratorAgent** – plan-and-dispatch + aggregate-results

---

### Task 5: Core Engine – Dynamic Skill Registration & Dual-layer Routing

**Files:**
- Create: `src/engine/skill_router/__init__.py`
- Create: `src/engine/skill_router/vector_ranker.py`
- Create: `src/engine/skill_router/llm_ranker.py`
- Create: `src/engine/skill_router/router.py`
- Create: `src/engine/skill_router/registry.py`

**Unit tests:** `tests/unit/test_skill_router.py`, `tests/unit/test_skill_registry.py`

- [ ] **Step 1: Implement VectorRanker** – cosine similarity ranking
- [ ] **Step 2: Implement LLMRanker** – LLM-based semantic reranking
- [ ] **Step 3: Implement SkillRouter** – vector → LLM dual-layer routing
- [ ] **Step 4: Implement SkillRegistry** – dynamic registration + search + auto-register from execution

---

### Task 6: Core Engine – Prompt Cache + Hierarchical Context Cache

**Files:**
- Create: `src/engine/prompt_cache/__init__.py`
- Create: `src/engine/prompt_cache/cache_manager.py`
- Create: `src/engine/prompt_cache/content_addressed.py`
- Create: `src/engine/context_compressor/__init__.py`
- Create: `src/engine/context_compressor/hierarchical.py`
- Create: `src/engine/context_compressor/summarizer.py`
- Create: `src/engine/context_compressor/decompressor.py`

**Unit tests:** `tests/unit/test_prompt_cache.py`, `tests/unit/test_context_compressor.py`

- [ ] **Step 1: Implement ContentAddressedCache** – SHA-256 key, TTL support
- [ ] **Step 2: Implement PromptCacheManager** – get-or-compute with cache-aside pattern
- [ ] **Step 3: Implement Summarizer** – chunk + LLM-summarize
- [ ] **Step 4: Implement HierarchicalCompressor** – compress context tree
- [ ] **Step 5: Implement Decompressor** – lazy decompression by chunk

---

### Task 7: Core Engine – Self-evolving Memory

**Files:**
- Create: `src/engine/memory/__init__.py`
- Create: `src/engine/memory/episodic.py`
- Create: `src/engine/memory/procedural.py`
- Create: `src/engine/memory/consolidator.py`

**Unit test:** `tests/unit/test_memory.py`

- [ ] **Step 1: Implement EpisodicMemory** – record task episodes
- [ ] **Step 2: Implement ProceduralMemory** – extract patterns from episodes
- [ ] **Step 3: Implement MemoryConsolidator** – offline reflection + pattern extraction

---

### Task 8: Core Engine – Multi-layer Safety Review

**Files:**
- Create: `src/engine/safety/__init__.py`
- Create: `src/engine/safety/human_in_loop.py`
- Create: `src/engine/safety/validator.py`

**Unit test:** `tests/unit/test_safety.py`

- [ ] **Step 1: Implement PatchValidator** – regex + AST-based security checks
- [ ] **Step 2: Implement AIRiskClassifier** – LLM-based risk assessment
- [ ] **Step 3: Implement HumanInLoop** – 3-layer approval flow

---

### Task 9: Agent Nodes

**Files:**
- Create: `src/orchestration/__init__.py`
- Create: `src/orchestration/graph.py`
- Create: `src/orchestration/react_loop.py`
- Create: `src/orchestration/nodes/__init__.py`
- Create: `src/orchestration/nodes/intent.py`
- Create: `src/orchestration/nodes/plan.py`
- Create: `src/orchestration/nodes/execute.py`
- Create: `src/orchestration/nodes/verify.py`

- [ ] **Step 1: Create LangGraph StateGraph – define 4-phase pipeline**
- [ ] **Step 2: Implement IntentNode** – intent recognition via LLM
- [ ] **Step 3: Implement PlanningNode** – decompose intent into execution plan
- [ ] **Step 4: Implement ExecutionNode** – dispatch sub-agents via orchestrator
- [ ] **Step 5: Implement VerificationNode** – run tests, collect results
- [ ] **Step 6: Implement react_loop** – ReAct-style tool-use loop

---

### Task 10: Sub-Agents

**Files:**
- Create: `src/agents/__init__.py`
- Create: `src/agents/bug_locator.py`
- Create: `src/agents/patch_generator.py`
- Create: `src/agents/test_verifier.py`

- [ ] **Step 1: Implement BugLocatorAgent** – read files, analyze, report bugs
- [ ] **Step 2: Implement PatchGeneratorAgent** – generate patches for identified bugs
- [ ] **Step 3: Implement TestVerifierAgent** – run tests, validate patches

---

### Task 11: Pipeline Integration

- [ ] **Step 1: Wire MiniCodeGraph** – connect all 4 nodes in LangGraph
- [ ] **Step 2: Implement run_pipeline** – main entry point (also in main.py)
- [ ] **Step 3: Create sample project** – calculator sample with a zero-division bug

---

### Task 12: Testing

- [ ] **Step: Verify all tests pass**

```bash
# unit tests
pytest tests/unit/ -v

# integration tests
pytest tests/integration/ -v

# e2e tests
pytest tests/e2e/ -v

# full suite with coverage
pytest --cov=src -v
```
