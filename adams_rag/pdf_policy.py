"""Shared, bounded file allowances for PDF download, parsing and OCR."""
import hashlib
from pathlib import Path

DEFAULT_PDF_BYTES = 25 * 1024 * 1024
MAX_PDF_BYTES = 512 * 1024 * 1024


def validate_pdf_bytes(value):
    if type(value) is not int or not 1 <= value <= MAX_PDF_BYTES:
        raise ValueError("PDF file allowance must be positive and no greater than 512 MiB")
    return value


def file_sha256(path, max_bytes=DEFAULT_PDF_BYTES):
    """Hash incrementally; a growing source cannot evade the file allowance."""
    validate_pdf_bytes(max_bytes)
    checksum, count = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        while block := stream.read(min(1024 * 1024, max_bytes + 1 - count)):
            count += len(block)
            if count > max_bytes:
                raise ValueError("PDF exceeds the configured file allowance")
            checksum.update(block)
    return checksum.hexdigest()
