"""Structured logging helpers."""
import json
import logging
import os
from typing import Any

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# Silence noisy third-party loggers like neo4j notifications
logging.getLogger("neo4j").setLevel(max(logging.WARNING, getattr(logging, LOG_LEVEL, logging.INFO)))
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
logging.getLogger("neo4j").propagate = False
logging.getLogger("neo4j.notifications").propagate = False

LOGGER = logging.getLogger("mcp_rag")


def _log(level: int, tool: str, event: str, **fields: Any) -> None:
    payload = {"tool": tool, "event": event, **fields}
    LOGGER.log(level, json.dumps(payload, ensure_ascii=False))


def log_info(tool: str, event: str, **fields: Any) -> None:
    _log(logging.INFO, tool, event, **fields)


def log_warning(tool: str, event: str, **fields: Any) -> None:
    _log(logging.WARNING, tool, event, **fields)


def log_error(tool: str, event: str, **fields: Any) -> None:
    _log(logging.ERROR, tool, event, **fields)
