"""Recognize one PDF page locally; the supervising process enforces resource limits."""
import contextlib
import json
import math
import os
import platform
from pathlib import Path
import sys
import threading

from .model_worker import watch_parent
from .ocr import POLICY_VERSION
from .pdf_policy import DEFAULT_PDF_BYTES, validate_pdf_bytes


def recognize(payload):
    path = Path(payload["path"]).resolve(strict=True)
    number = payload["page"]
    max_bytes = validate_pdf_bytes(payload.get("max_bytes", DEFAULT_PDF_BYTES))
    if path.stat().st_size > max_bytes or type(number) is not int or not 1 <= number <= 5000:
        raise ValueError("OCR file or page allowance exceeded")
    import Foundation
    import Quartz as Q
    import Vision

    pdf = Q.CGPDFDocumentCreateWithURL(Foundation.NSURL.fileURLWithPath_(str(path)))
    if pdf is None or number > Q.CGPDFDocumentGetNumberOfPages(pdf):
        raise ValueError("PDF page unavailable")
    page = Q.CGPDFDocumentGetPage(pdf, number)
    bounds = Q.CGPDFPageGetBoxRect(page, Q.kCGPDFMediaBox)
    width, height = bounds.size.width, bounds.size.height
    if not all(math.isfinite(x) and x > 0 for x in (width, height)):
        raise ValueError("Invalid PDF page dimensions")
    if Q.CGPDFPageGetRotationAngle(page) % 180:
        width, height = height, width
    scale = min(2200 / max(width, height), 3)
    width, height = max(1, round(width * scale)), max(1, round(height * scale))
    rect = Q.CGRectMake(0, 0, width, height)
    context = Q.CGBitmapContextCreate(None, width, height, 8, width * 4,
                                    Q.CGColorSpaceCreateDeviceRGB(), Q.kCGImageAlphaPremultipliedLast)
    if context is None:
        raise RuntimeError("PDF rasterization unavailable")
    Q.CGContextSetRGBFillColor(context, 1, 1, 1, 1)
    Q.CGContextFillRect(context, rect)
    # Quartz's PDF transform may decline to enlarge a small page. Apply the raster
    # scale explicitly; use page-sized coordinates for its rotation/centering step.
    Q.CGContextScaleCTM(context, scale, scale)
    transform = Q.CGPDFPageGetDrawingTransform(page, Q.kCGPDFMediaBox,
                                             Q.CGRectMake(0, 0, width / scale, height / scale), 0, True)
    Q.CGContextConcatCTM(context, transform)
    Q.CGContextDrawPDFPage(context, page)
    image = Q.CGBitmapContextCreateImage(context)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRevision_(3)
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setRecognitionLanguages_(["en-US"])
    request.setUsesLanguageCorrection_(False)
    # Bound CPU/memory pressure and avoid depending on GPU availability.
    request.setUsesCPUOnly_(True)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, {})
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError("Local OCR failed")
    lines = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        candidate = candidates[0]
        box = observation.boundingBox()
        lines.append({"text": str(candidate.string()), "confidence": float(candidate.confidence()),
                      "box": [box.origin.x, box.origin.y, box.size.width, box.size.height]})
    # Vision's observation order is retained. Neither it nor these boxes guarantee
    # correct table/multicolumn reading order; the original page remains authoritative.
    return {"text": "\n".join(line["text"] for line in lines), "lines": lines,
            "page": number, "method": "apple_vision_ocr", "revision": int(request.revision()),
            "extraction_policy_version": POLICY_VERSION, "macos_version": platform.mac_ver()[0],
            "language_correction": False, "language": "en-US", "raster_pixels": [width, height],
            "confidence_is_accuracy": False, "layout_verified": False}


def main():
    try:
        payload = json.loads(sys.stdin.buffer.read(65537))
        if type(payload.get("parent_pid")) is not int or payload["parent_pid"] != os.getppid():
            raise ValueError("Supervisor unavailable")
        threading.Thread(target=watch_parent, args=(payload["parent_pid"],), daemon=True).start()
        with contextlib.redirect_stdout(sys.stderr):
            result = recognize(payload)
        print(json.dumps({"ok": True, "result": result}), flush=True)
    except Exception:
        print(json.dumps({"ok": False}), flush=True)


if __name__ == "__main__":
    main()
