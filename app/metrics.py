"""
Prometheus metrics definitions shared across the application.
"""

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "siglip_requests_total",
    "Total HTTP requests",
    ["route", "status"],
)

REQUEST_LATENCY = Histogram(
    "siglip_request_duration_seconds",
    "HTTP request duration in seconds",
    ["route"],
)

PIPELINE_STAGE_LATENCY = Histogram(
    "siglip_pipeline_stage_duration_seconds",
    "Per-stage pipeline latency",
    ["stage"],
)

DECISIONS_TOTAL = Counter(
    "siglip_decisions_total",
    "Decision counts by action",
    ["action"],
)

TEI_ERRORS_TOTAL = Counter(
    "siglip_tei_errors_total",
    "TEI call failures",
    ["service"],   # "embed" or "rerank"
)

DB_ERRORS_TOTAL = Counter(
    "siglip_db_errors_total",
    "Database call failures",
)

