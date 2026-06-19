# MiniCode — AI-Powered Automated Code Repair Agent

[![CI](https://github.com/Lordyuyu/minicode/actions/workflows/ci.yml/badge.svg)](https://github.com/Lordyuyu/minicode/actions/workflows/ci.yml)

MiniCode 是一个基于 LangGraph + LLM 的自动化代码修复代理。输入一个带 Bug 的 Python 代码库，自动完成 Bug 定位 → 补丁生成 → 测试验证 → 安全审查的完整闭环。

## Benchmark

在 7 类典型 Python Bug 上的表现（DeepSeek Chat）：

| 指标 | 数值 |
|---|---|
| Bug 定位准确率 | **100%** |
| 补丁生成成功率 | **100%** |
| 端到端修复率 | **100%** |

```
PASS  divide-by-zero              15.9s
PASS  index-out-of-range          17.9s
PASS  none-attribute-access       17.4s
PASS  key-error-missing            8.5s
PASS  type-error-concatenation     9.4s
PASS  mutable-default-arg         10.5s
PASS  infinite-recursion          12.6s
────────────────────────────────────────
7 passed, 0 failed
```

## Architecture

```
CLI / API
   │
   ▼
┌─────────────────────────────────────────┐
│            LangGraph Orchestrator        │
│  intent ──► plan ──► execute ──► verify │
│    ▲                            │        │
│    └────── loop (≤3 iter) ──────┘        │
└─────────────────────────────────────────┘
   │                │              │
   ▼                ▼              ▼
┌──────────┐ ┌────────────┐ ┌───────────┐
│ BugLocator│ │PatchGenerator│ │ReAct Loop │
│  (LLM)    │ │   (LLM)     │ │ (4 tools) │
└──────────┘ └────────────┘ └───────────┘
   │                │              │
   └────────────────┼──────────────┘
                    ▼
┌─────────────────────────────────────────┐
│           Core Engine                    │
│  Skill Router │ Context Compressor       │
│  Prompt Cache │ Memory (3-tier)          │
│  Safety (3-layer) │ Permission Ctrl      │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│          Infrastructure                  │
│  PostgreSQL + pgvector │ Redis │ DeepSeek│
└─────────────────────────────────────────┘
```

### Pipeline (4 phases)

1. **Intent** — LLM 识别意图类型（bug_fix / feature_add / refactor / dependency_fix）
2. **Plan** — 分解为执行步骤
3. **Execute** — 运行测试 → BugLocator 定位 → PatchGenerator 并发生成补丁
4. **Verify** — 运行 pytest 验证，失败则回到 Intent（最多 3 轮）

### Core Modules

| 模块 | 功能 |
|---|---|
| `agents/` | BugLocator（错误定位）、PatchGenerator（补丁生成，支持并发） |
| `orchestration/` | LangGraph 状态图、ReAct 工具调用循环、4 个节点 |
| `engine/skill_router/` | 向量相似度 + LLM 重排序双层技能路由，自动注册成功模式 |
| `engine/context_compressor/` | 层级压缩，Redis 外存，递归深度限制 |
| `engine/memory/` | 三级记忆：情景（轨迹）→ 过程（模式）→ 语义（向量检索） |
| `engine/safety/` | 三层审查：35+ 正则 → AI 风险分类 → Human-in-the-Loop |
| `llm/` | DeepSeek 客户端（指数退避重试、Token 日志）、Few-shot+CoT 提示 |
| `storage/` | PostgreSQL+pgvector（HNSW 向量索引）、Redis（缓存+分块存储） |
| `api/` | FastAPI + SSE 流式推送 |
| `evaluation/` | 7 场景 Benchmark 框架 |
| `observability/` | LangSmith 集成 |

## Quick Start

### Requirements

- Python ≥ 3.12
- PostgreSQL + pgvector（可选，无 Redis/DB 也能跑，功能降级）
- DeepSeek API Key

### Install

```bash
git clone <repo-url> && cd minicode
pip install -e .
```

### Configure

```bash
cp .env.example .env
# Edit .env — set DEEPSEEK_API_KEY
```

### CLI

```bash
# 修复一个代码库
python -m src.main /path/to/buggy/project --test "pytest -x"
```

### API

```bash
uvicorn src.api.server:app --reload

# Start a repair task
curl -X POST http://localhost:8000/api/v1/repair \
  -H "Content-Type: application/json" \
  -d '{"codebase_path": "/path/to/project", "test_command": "pytest"}'

# Stream progress (SSE)
curl -N http://localhost:8000/api/v1/repair/{task_id}/stream
```

### Evaluation

```bash
# Run full benchmark
python -m src.evaluation.benchmark

# List scenarios
python -m src.evaluation.benchmark --list

# Filter by tag
python -m src.evaluation.benchmark --tags arithmetic recursion
```

### Tests

```bash
pytest tests/ -v    # 43 tests
```

## Tech Stack

`LangGraph` `DeepSeek` `PostgreSQL` `pgvector` `Redis` `FastAPI` `SSE` `LangSmith` `OpenAI SDK` `SQLAlchemy` `Pydantic` `Loguru` `NumPy`
