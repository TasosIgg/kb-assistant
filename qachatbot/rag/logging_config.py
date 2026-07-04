# rag/logging_config.py

import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        return json.dumps(payload)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("kb_assistant")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


def log_request(logger: logging.Logger, *, question: str, answered: bool,
                 latency_ms: int, top_source: str | None, flagged: bool = False):
    logger.info(
        "kb_query",
        extra={"extra_fields": {
            "question": question,
            "answered": answered,
            "flagged_injection": flagged,
            "latency_ms": latency_ms,
            "top_source": top_source,
        }},
    )
