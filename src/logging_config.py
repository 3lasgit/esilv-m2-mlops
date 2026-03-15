# ============================================================
# src/logging_config.py
# Configuration du logging structuré (JSON) pour la production
# ============================================================
# Usage :
#   from logging_config import setup_logging
#   setup_logging()            # → JSON en production
#   setup_logging(dev=True)    # → format lisible en dev
# ============================================================

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """
    Formatte les logs en JSON structuré — compatible ELK / CloudWatch / Datadog.

    Champs émis : timestamp, level, logger, message, module, funcName, lineno.
    Les extras passés via ``extra={}`` sont automatiquement ajoutés.
    """

    EXCLUDE_KEYS = {
        "name", "msg", "args", "created", "relativeCreated",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "filename", "module", "levelno", "levelname", "pathname",
        "thread", "threadName", "processName", "process", "message",
        "msecs", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Ajoute les extras (ex. request_id, latency_ms, prediction)
        for key, value in record.__dict__.items():
            if key not in self.EXCLUDE_KEYS:
                log_entry[key] = value

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str, ensure_ascii=False)


def setup_logging(dev: bool | None = None) -> None:
    """
    Configure le logging pour l'application.

    Parameters
    ----------
    dev : bool | None
        True  → format lisible (console dev)
        False → JSON structuré (production)
        None  → détection automatique via ENV (défaut : production)
    """
    if dev is None:
        env = os.getenv("APP_ENV", "production").lower()
        dev = env in ("dev", "development", "local")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Supprime les handlers existants pour éviter les doublons
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if dev:
        fmt = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    else:
        handler.setFormatter(JSONFormatter())

    root.addHandler(handler)

    # Réduit le bruit des libs tierces
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
