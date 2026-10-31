"""
Secure ZIP file validation and extraction for the evaluation pipeline.
Guards against zip-slip (path traversal), zip bombs (size/count limits),
and unsupported file types.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings

logger = logging.getLogger("siglip.eval.zip")

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ZipValidationError(Exception):
    """Raised when an uploaded ZIP violates security or format constraints."""
    pass


@dataclass
class ExtractedDataset:
    temp_dir: str
    csv_path: str
    image_paths: dict[str, str]  # normalized relative name -> full local path
    total_images: int
    uncompressed_bytes: int


def inspect_and_extract_zip(
    zip_bytes: bytes,
    settings: Settings,
) -> ExtractedDataset:
    """
    Validates ZIP file safety in-memory, then extracts to a dedicated temp directory.
    Enforces zip-slip prevention, total uncompressed size cap, file count cap,
    per-image size cap, and image file type constraints.
    """
    if len(zip_bytes) > settings.EVAL_MAX_ZIP_BYTES:
        raise ZipValidationError(
            f"Uploaded ZIP size ({len(zip_bytes)} bytes) exceeds maximum "
            f"limit of {settings.EVAL_MAX_ZIP_BYTES} bytes."
        )

    import io
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as exc:
        raise ZipValidationError(f"Invalid or corrupted ZIP file: {exc}") from exc

    namelist = zf.namelist()
    if not namelist:
        raise ZipValidationError("The uploaded ZIP archive is completely empty.")

    # Check file count
    if len(namelist) > settings.EVAL_MAX_IMAGE_COUNT + 10:
        raise ZipValidationError(
            f"ZIP contains {len(namelist)} entries, exceeding maximum limit "
            f"of {settings.EVAL_MAX_IMAGE_COUNT} files."
        )

    total_uncompressed = 0
    csv_member: str | None = None

    # First pass: inspect headers and validate safety
    for info in zf.infolist():
        # Skip directory entries
        if info.is_dir():
            continue

        filename = info.filename
        norm_name = os.path.normpath(filename).replace("\\", "/")

        # Zip-slip check
        if norm_name.startswith("/") or ".." in norm_name or ":" in norm_name:
            raise ZipValidationError(
                f"Path traversal detected in ZIP entry: '{filename}'"
            )

        # Cap per-file size
        total_uncompressed += info.file_size
        if info.file_size > settings.EVAL_MAX_IMAGE_BYTES and not norm_name.lower().endswith(".csv"):
            raise ZipValidationError(
                f"File '{filename}' ({info.file_size} bytes) exceeds per-image limit "
                f"of {settings.EVAL_MAX_IMAGE_BYTES} bytes."
            )

        # Extension check
        ext = os.path.splitext(norm_name)[1].lower()
        base_name = os.path.basename(norm_name).lower()

        if base_name == "ground_truth.csv" or ext == ".csv":
            if csv_member is None or base_name == "ground_truth.csv":
                csv_member = filename
        elif ext in ALLOWED_IMAGE_EXTENSIONS:
            pass  # Allowed image
        elif base_name in {".ds_store", "thumbs.db"}:
            pass  # Ignore OS metadata files
        else:
            raise ZipValidationError(
                f"Unsupported file type in ZIP: '{filename}'. "
                f"Only JPG, JPEG, PNG, WebP images and ground_truth.csv are accepted."
            )

    if total_uncompressed > settings.EVAL_MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ZipValidationError(
            f"Total uncompressed size ({total_uncompressed} bytes) exceeds "
            f"limit of {settings.EVAL_MAX_TOTAL_UNCOMPRESSED_BYTES} bytes."
        )

    if not csv_member:
        raise ZipValidationError(
            "Missing ground_truth.csv in the uploaded ZIP. "
            "The archive must contain a ground_truth.csv file."
        )

    # Create temporary extraction directory
    temp_dir = tempfile.mkdtemp(prefix="siglip_eval_")
    real_temp_dir = os.path.realpath(temp_dir)

    image_paths: dict[str, str] = {}
    extracted_csv_path: str = ""

    try:
        for info in zf.infolist():
            if info.is_dir():
                continue

            rel_path = os.path.normpath(info.filename).replace("\\", "/")
            dest_path = os.path.realpath(os.path.join(real_temp_dir, rel_path))

            # Verify destination stays strictly within temp_dir
            if not dest_path.startswith(real_temp_dir + os.sep) and dest_path != real_temp_dir:
                raise ZipValidationError(f"Extraction path escape detected: '{info.filename}'")

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with zf.open(info) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            ext = os.path.splitext(rel_path)[1].lower()
            base_lower = os.path.basename(rel_path).lower()

            if info.filename == csv_member:
                extracted_csv_path = dest_path

            if ext in ALLOWED_IMAGE_EXTENSIONS:
                # Store multiple lookups: exact relative path, basename, and photos/ prefix
                image_paths[rel_path] = dest_path
                image_paths[os.path.basename(rel_path)] = dest_path
                # Normalize forward slash
                image_paths[rel_path.lower()] = dest_path
                image_paths[os.path.basename(rel_path).lower()] = dest_path

        if not extracted_csv_path or not os.path.exists(extracted_csv_path):
            raise ZipValidationError("Failed to extract ground_truth.csv.")

        return ExtractedDataset(
            temp_dir=temp_dir,
            csv_path=extracted_csv_path,
            image_paths=image_paths,
            total_images=len([k for k in image_paths if "/" not in k and "\\" not in k]),
            uncompressed_bytes=total_uncompressed,
        )

    except Exception:
        cleanup_temp_dir(temp_dir)
        raise


def resolve_image_path(dataset: ExtractedDataset, filename: str) -> str | None:
    """
    Resolve an image filename from CSV to its extracted local file path.
    Handles direct filenames, photos/ subfolders, and case-insensitivity.
    """
    clean_name = filename.strip().replace("\\", "/")
    if not clean_name:
        return None

    # 1. Exact match
    if clean_name in dataset.image_paths:
        return dataset.image_paths[clean_name]

    # 2. Case-insensitive exact match
    if clean_name.lower() in dataset.image_paths:
        return dataset.image_paths[clean_name.lower()]

    # 3. Basename match
    base = os.path.basename(clean_name)
    if base in dataset.image_paths:
        return dataset.image_paths[base]
    if base.lower() in dataset.image_paths:
        return dataset.image_paths[base.lower()]

    # 4. Strip leading photos/ or add photos/
    if clean_name.lower().startswith("photos/"):
        stripped = clean_name[7:]
        if stripped in dataset.image_paths:
            return dataset.image_paths[stripped]
        if stripped.lower() in dataset.image_paths:
            return dataset.image_paths[stripped.lower()]
    else:
        with_photos = f"photos/{clean_name}"
        if with_photos in dataset.image_paths:
            return dataset.image_paths[with_photos]
        if with_photos.lower() in dataset.image_paths:
            return dataset.image_paths[with_photos.lower()]

    return None


def cleanup_temp_dir(temp_dir: str | None) -> None:
    """Safely delete extracted temporary directory."""
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("Cleaned up eval temp dir: %s", temp_dir)
        except Exception as exc:
            logger.warning("Failed to clean up temp dir %s: %s", temp_dir, exc)

