"""
Validation of evaluation ground_truth.csv and image assets.
Generates pre-run validation reports before pipeline execution.
"""

from __future__ import annotations

import csv
import difflib
import io
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from PIL import Image

from app.eval.zip_handler import ExtractedDataset, resolve_image_path

logger = logging.getLogger("siglip.eval.validator")


@dataclass
class EvalRowInput:
    row_id: int
    image_name: str
    image_path: str | None
    description: str
    expected_service: str
    is_none: bool
    is_empty_desc: bool


@dataclass
class ValidationReport:
    total_rows: int = 0
    valid_rows_count: int = 0
    missing_images: list[dict[str, Any]] = field(default_factory=list)
    corrupt_images: list[dict[str, Any]] = field(default_factory=list)
    unknown_services: list[dict[str, Any]] = field(default_factory=list)
    duplicate_rows: list[dict[str, Any]] = field(default_factory=list)
    service_counts: dict[str, int] = field(default_factory=dict)
    empty_description_count: int = 0
    none_count: int = 0
    has_critical_errors: bool = False
    error_messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Validated rows ready for execution
    valid_rows: list[EvalRowInput] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Exclude internal valid_rows from lightweight UI report
        d.pop("valid_rows", None)
        return d


def validate_ground_truth(
    dataset: ExtractedDataset,
    active_service_titles: list[str],
) -> ValidationReport:
    """
    Parses ground_truth.csv and inspects all image files and service labels.
    Suggests closest valid WoService titles for typos or mismatches.
    """
    report = ValidationReport()
    active_set = set(active_service_titles)
    # Case-insensitive mapping for suggestions
    title_lookup = {t.lower(): t for t in active_service_titles}

    if not os.path.exists(dataset.csv_path):
        report.has_critical_errors = True
        report.error_messages.append("ground_truth.csv was not found in dataset.")
        return report

    try:
        with open(dataset.csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = [c.strip().lower() for c in (reader.fieldnames or [])]

            if "image" not in fieldnames or "service" not in fieldnames:
                report.has_critical_errors = True
                report.error_messages.append(
                    f"ground_truth.csv headers must include 'image' and 'service'. "
                    f"Found headers: {reader.fieldnames}"
                )
                return report

            # Find actual casing for key lookups
            orig_fields = reader.fieldnames or []
            img_key = next(k for k in orig_fields if k.strip().lower() == "image")
            svc_key = next(k for k in orig_fields if k.strip().lower() == "service")
            desc_key = next((k for k in orig_fields if k.strip().lower() == "description"), None)

            seen_exact_rows: set[tuple[str, str, str]] = set()

            for row_idx, row in enumerate(reader, start=1):
                report.total_rows += 1
                raw_img = (row.get(img_key) or "").strip()
                raw_svc = (row.get(svc_key) or "").strip()
                raw_desc = (row.get(desc_key) or "").strip() if desc_key else ""

                row_key = (raw_img.lower(), raw_desc.lower(), raw_svc.lower())
                if row_key in seen_exact_rows:
                    report.duplicate_rows.append({
                        "row": row_idx,
                        "image": raw_img,
                        "description": raw_desc,
                        "service": raw_svc,
                        "reason": "Duplicate row (identical image, description, and service)",
                    })
                seen_exact_rows.add(row_key)

                # Parse service label: NONE / none / None (any case, trimmed) -> NONE class
                is_none = (raw_svc.strip().upper() == "NONE")
                if is_none:
                    report.none_count += 1
                    effective_service = "NONE"
                else:
                    # Check against active services
                    if raw_svc in active_set:
                        effective_service = raw_svc
                        report.service_counts[raw_svc] = report.service_counts.get(raw_svc, 0) + 1
                    elif raw_svc.lower() in title_lookup:
                        # Exact match under different casing
                        canonical = title_lookup[raw_svc.lower()]
                        effective_service = canonical
                        report.service_counts[canonical] = report.service_counts.get(canonical, 0) + 1
                        report.warnings.append(
                            f"Row {row_idx}: Service casing '{raw_svc}' normalized to '{canonical}'."
                        )
                    else:
                        # Unmatched service: suggest closest title
                        suggestions = difflib.get_close_matches(
                            raw_svc, active_service_titles, n=2, cutoff=0.35
                        )
                        suggestion_str = suggestions[0] if suggestions else None
                        report.unknown_services.append({
                            "row": row_idx,
                            "service": raw_svc,
                            "suggestion": suggestion_str,
                        })
                        effective_service = raw_svc
                        report.service_counts[raw_svc] = report.service_counts.get(raw_svc, 0) + 1

                is_empty_desc = not bool(raw_desc)
                if is_empty_desc:
                    report.empty_description_count += 1

                # Check image existence
                img_path = resolve_image_path(dataset, raw_img)
                if not img_path or not os.path.exists(img_path):
                    report.missing_images.append({
                        "row": row_idx,
                        "image": raw_img,
                    })
                    img_path = None
                else:
                    # Test image readability
                    try:
                        with Image.open(img_path) as im:
                            im.verify()
                    except Exception as img_err:
                        report.corrupt_images.append({
                            "row": row_idx,
                            "image": raw_img,
                            "error": str(img_err),
                        })
                        img_path = None

                # Build row input
                report.valid_rows.append(
                    EvalRowInput(
                        row_id=row_idx,
                        image_name=raw_img,
                        image_path=img_path,
                        description=raw_desc,
                        expected_service=effective_service,
                        is_none=is_none,
                        is_empty_desc=is_empty_desc,
                    )
                )

        # Count usable rows
        usable = [
            r for r in report.valid_rows
            if r.image_path is not None
        ]
        report.valid_rows_count = len(usable)

        if report.total_rows == 0:
            report.has_critical_errors = True
            report.error_messages.append("ground_truth.csv contains 0 data rows.")
        elif report.valid_rows_count == 0:
            report.has_critical_errors = True
            report.error_messages.append("None of the rows have readable images found in the ZIP.")

        if report.unknown_services:
            report.warnings.append(
                f"{len(report.unknown_services)} rows refer to unknown service names "
                f"not matching active WoService titles."
            )

        if report.missing_images:
            report.warnings.append(
                f"{len(report.missing_images)} rows reference images that were missing from the ZIP."
            )

        if report.corrupt_images:
            report.warnings.append(
                f"{len(report.corrupt_images)} images could not be decoded or are corrupt."
            )

    except Exception as exc:
        logger.error("Failed to parse ground_truth.csv: %s", exc, exc_info=True)
        report.has_critical_errors = True
        report.error_messages.append(f"Failed to read ground_truth.csv: {exc}")

    return report

