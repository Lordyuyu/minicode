from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.observability.tracer import setup_langsmith
from src.orchestration.graph import MiniCodeGraph
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


async def run_pipeline(codebase_path: str, test_command: str) -> None:
    setup_logging()
    setup_langsmith()
    logger.info("Starting MiniCode pipeline for {}", codebase_path)

    graph = MiniCodeGraph()
    result = await graph.run(
        input_codebase_path=codebase_path,
        test_command=test_command,
    )

    if result.pipeline_success:
        logger.info("Pipeline completed successfully")
        for i, patch in enumerate(result.patches):
            logger.info("  Patch {}: {} (verified: {})", i + 1, patch.file_path, patch.verified)
    else:
        logger.error("Pipeline failed: {}", result.errors)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniCode - AI Coding Agent")
    parser.add_argument("codebase", type=str, help="Path to Python codebase")
    parser.add_argument("--test", "-t", type=str, default="pytest", help="Test command")
    args = parser.parse_args()

    codebase_path = str(Path(args.codebase).resolve())
    asyncio.run(run_pipeline(codebase_path, args.test))


if __name__ == "__main__":
    main()
