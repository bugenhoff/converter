"""Groq LLM-based PDF to DOCX conversion with vision models."""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

try:
    import groq
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.shared import OxmlElement, qn
    from PIL import Image
    from pdf2image import convert_from_bytes
except ImportError:
    groq = None
    Document = None
    convert_from_bytes = None

from ..config.settings import settings

logger = logging.getLogger(__name__)
T = TypeVar("T")
_DEFAULT_MIN_TEXT_CHARS_PER_PAGE = 20


class GroqConversionError(RuntimeError):
    """Signals that Groq LLM conversion failed."""


@dataclass
class _GroqTask:
    page_items: list[tuple[int, T]]
    image_max_side: int
    retries_left: int


def convert_pdf_to_docx_via_groq(source_path: Path, output_dir: Path) -> Path:
    """Convert PDF to DOCX using Groq vision LLM."""
    
    if not groq or not Document:
        raise ImportError("groq and python-docx are required for LLM conversion")
    
    if not settings.groq_api_key:
        raise GroqConversionError("GROQ_API_KEY is not configured")

    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file {source_path} was not found")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docx_path = output_dir / f"{source_path.stem}.docx"

    logger.info("Converting PDF to images for LLM processing: %s", source_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert PDF to images
        images = _pdf_to_images(source_path, Path(temp_dir))
        
        if not images:
            raise GroqConversionError("Failed to extract images from PDF")
        
        logger.info("Extracted %d page images, sending to Groq LLM", len(images))
        
        # Process with Groq LLM
        structured_content = _process_images_with_groq(images)
        
        # Create DOCX from structured content  
        _create_docx_from_content(structured_content, docx_path)
        
        logger.info("LLM-based conversion completed: %s", docx_path)

    if not docx_path.exists():
        raise GroqConversionError("Output file is missing after conversion")
    
    if docx_path.stat().st_size == 0:
        raise GroqConversionError("Conversion produced an empty DOCX file")

    return docx_path


def convert_pdf_bytes_to_docx_via_groq(pdf_bytes: bytes, original_name: str) -> bytes:
    """Convert PDF bytes to DOCX bytes using Groq vision LLM (in-memory processing)."""
    
    if not groq or not Document or not convert_from_bytes:
        raise ImportError("groq, python-docx, and pdf2image are required for memory conversion")
    
    if not settings.groq_api_key:
        raise GroqConversionError("GROQ_API_KEY is not configured")
    
    logger.info("Converting PDF bytes to DOCX in memory: %s (%d bytes)", original_name, len(pdf_bytes))
    logger.info(
        "Groq config: model=%s max_tokens=%d batch_size=%d max_side=%d dpi=%d",
        settings.groq_model,
        settings.groq_max_tokens,
        settings.groq_batch_size,
        settings.groq_image_max_side,
        settings.groq_pdf_image_dpi,
    )
    
    try:
        # Convert PDF bytes to images in memory
        images = _pdf_bytes_to_images(pdf_bytes)
        
        if not images:
            raise GroqConversionError("Failed to extract images from PDF bytes")
        
        logger.info("Extracted %d page images from memory, sending to Groq LLM", len(images))
        
        # Process with Groq LLM
        structured_content = _process_pil_images_with_groq(images)
        
        # Create DOCX in memory
        docx_bytes = _create_docx_bytes_from_content(structured_content)
        
        logger.info("Memory-based LLM conversion completed for %s: %d bytes", 
                   original_name, len(docx_bytes))
        
        return docx_bytes
        
    except Exception as e:
        raise GroqConversionError(f"Memory conversion failed for {original_name}: {e}") from e


def _pdf_bytes_to_images(pdf_bytes: bytes) -> list[Image.Image]:
    """Convert PDF bytes to PIL Images in memory."""
    try:
        # Use pdf2image to convert bytes directly to PIL Images
        images = convert_from_bytes(
            pdf_bytes,
            dpi=settings.groq_pdf_image_dpi,
            fmt='PNG',
            use_pdftocairo=True  # Better quality rendering
        )
        logger.debug(
            "Converted PDF bytes to %d PIL images (dpi=%d)",
            len(images),
            settings.groq_pdf_image_dpi,
        )
        return images
        
    except Exception as e:
        raise GroqConversionError(f"Failed to convert PDF bytes to images: {e}") from e


def _process_pil_images_with_groq(images: list[Image.Image]) -> dict[str, Any]:
    """Process PIL images with adaptive Groq batching strategy."""
    merged = _process_images_with_adaptive_strategy(
        images,
        _process_pil_batch_with_groq,
    )
    _validate_merged_pages(merged, expected_pages=len(images))
    return merged


def _process_images_with_adaptive_strategy(
    images: list[T],
    processor: Callable[[Any, list[T], int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    """Process pages with adaptive split/downscale/retry strategy."""

    if not images:
        raise GroqConversionError("No images to process")

    client = groq.Groq(api_key=settings.groq_api_key)
    tasks = _build_initial_tasks(images)

    all_results: list[dict[str, Any]] = []
    max_requests = settings.groq_max_requests_per_document
    requests_made = 0

    logger.info(
        "Adaptive Groq scheduler started: pages=%d initial_batch=%d min_batch=%d max_side=%d min_side=%d max_requests=%s retries=%d",
        len(images),
        settings.groq_batch_size,
        settings.groq_min_batch_size,
        settings.groq_image_max_side,
        settings.groq_min_image_max_side,
        "unlimited" if max_requests == 0 else max_requests,
        settings.groq_retry_per_task,
    )

    while tasks:
        task = tasks.popleft()
        expected_pages = [page_number for page_number, _ in task.page_items]

        if max_requests > 0 and requests_made >= max_requests:
            raise GroqConversionError(
                f"Groq request limit reached ({requests_made}/{max_requests}) before full document coverage"
            )

        requests_made += 1
        request_cap = "unlimited" if max_requests == 0 else str(max_requests)
        logger.info(
            "Groq request %d/%s: pages=%s pages_in_task=%d image_max_side=%d max_tokens=%d",
            requests_made,
            request_cap,
            _format_page_numbers(expected_pages),
            len(expected_pages),
            task.image_max_side,
            settings.groq_max_tokens,
        )

        try:
            page_payload = [item for _, item in task.page_items]
            batch_result = processor(
                client,
                page_payload,
                expected_pages[0],
                requests_made - 1,
                task.image_max_side,
            )
            _validate_batch_pages(batch_result, expected_pages)
            all_results.append(batch_result)
            continue
        except Exception as exc:
            logger.warning(
                "Groq task failed for pages=%s image_max_side=%d retries_left=%d: %s",
                _format_page_numbers(expected_pages),
                task.image_max_side,
                task.retries_left,
                exc,
            )

        split_tasks = _split_task(task)
        if split_tasks:
            logger.info(
                "Adaptive action=split pages=%s into %s + %s",
                _format_page_numbers(expected_pages),
                _format_page_numbers([n for n, _ in split_tasks[0].page_items]),
                _format_page_numbers([n for n, _ in split_tasks[1].page_items]),
            )
            tasks.appendleft(split_tasks[1])
            tasks.appendleft(split_tasks[0])
            continue

        downscaled = _downscale_task(task)
        if downscaled:
            logger.info(
                "Adaptive action=downscale pages=%s old_side=%d new_side=%d",
                _format_page_numbers(expected_pages),
                task.image_max_side,
                downscaled.image_max_side,
            )
            tasks.appendleft(downscaled)
            continue

        if task.retries_left > 0:
            retried_task = _GroqTask(
                page_items=task.page_items,
                image_max_side=task.image_max_side,
                retries_left=task.retries_left - 1,
            )
            logger.info(
                "Adaptive action=retry pages=%s retries_left=%d",
                _format_page_numbers(expected_pages),
                retried_task.retries_left,
            )
            tasks.appendleft(retried_task)
            continue

        raise GroqConversionError(
            "Groq conversion failed after exhausting split/downscale/retry policy for pages "
            f"{_format_page_numbers(expected_pages)}"
        )

    if not all_results:
        raise GroqConversionError("All Groq tasks failed to process")

    merged = _merge_batch_results(all_results)
    logger.info(
        "Adaptive Groq scheduler completed: requests_made=%d final_pages=%d",
        requests_made,
        len(merged.get("pages", [])),
    )
    return merged


def _batch_expected_pages(start_page: int, batch_size: int) -> list[int]:
    return list(range(start_page, start_page + batch_size))


def _build_transcription_prompt(expected_pages: list[int]) -> str:
    page_numbers = ", ".join(str(page) for page in expected_pages)
    return f"""You are a strict OCR transcription engine.

Transcribe ALL visible text from each provided page.

Return ONLY valid JSON with this schema:
{{
  "pages": [
    {{"page_number": <int>, "text": "<full page text with \\n line breaks>"}}
  ]
}}

Rules:
1) Return exactly {len(expected_pages)} page objects.
2) page_number values must be exactly: [{page_numbers}].
3) Do not summarize, explain, translate, or omit text.
4) Keep original language and line order from the page.
5) Preserve line breaks inside "text".
6) Do not return Markdown, code fences, or any extra keys outside JSON.
"""


def _decode_groq_json_response(raw_content: str) -> dict[str, Any]:
    content = raw_content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GroqConversionError(f"Groq returned invalid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise GroqConversionError("Groq returned JSON that is not an object")
    return parsed


def _safe_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item is not None)
    if value is None:
        return ""
    return str(value)


def _normalized_text_length(text: str) -> int:
    return len("".join(text.split()))


def _extract_page_text(page: dict[str, Any]) -> str:
    direct_text = _safe_text(page.get("text", "")).strip()
    if direct_text:
        return direct_text

    direct_content = _safe_text(page.get("content", "")).strip()
    if direct_content:
        return direct_content

    sections = page.get("sections", [])
    if not isinstance(sections, list):
        return ""

    parts: list[str] = []
    for section in sections:
        if not isinstance(section, dict):
            continue

        content_text = _safe_text(section.get("content", "")).strip()
        if content_text:
            parts.append(content_text)

        table_data = section.get("table_data", {})
        if isinstance(table_data, dict):
            headers = table_data.get("headers", [])
            rows = table_data.get("rows", [])
            if isinstance(headers, list) and headers:
                parts.append(" | ".join(_safe_text(header).strip() for header in headers))
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, list):
                        row_text = " | ".join(_safe_text(cell).strip() for cell in row)
                        if row_text.strip():
                            parts.append(row_text)

    return "\n".join(part for part in parts if part.strip())


def _normalize_batch_result(result: dict[str, Any]) -> dict[str, Any]:
    raw_pages = result.get("pages", [])
    if not isinstance(raw_pages, list):
        raise GroqConversionError("Groq JSON does not contain a valid 'pages' list")

    normalized_pages: list[dict[str, Any]] = []
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            continue

        raw_page_number = raw_page.get("page_number")
        try:
            page_number = int(raw_page_number)
        except (TypeError, ValueError) as exc:
            raise GroqConversionError(
                f"Groq batch output has invalid page number: {raw_page_number!r}"
            ) from exc

        page_text = _extract_page_text(raw_page)
        normalized_page = dict(raw_page)
        normalized_page["page_number"] = page_number
        normalized_page["text"] = page_text
        normalized_pages.append(normalized_page)

    return {
        "title": _safe_text(result.get("title", "")),
        "pages": normalized_pages,
    }


def _batch_max_tokens(expected_pages: list[int]) -> int:
    estimated_tokens = 4000 * len(expected_pages)
    return min(settings.groq_max_tokens, max(2048, estimated_tokens))


def _process_pil_batch_with_groq(
    client,
    pil_images: list[Image.Image],
    start_page: int,
    batch_idx: int,
    image_max_side: int,
) -> dict[str, Any]:
    """Process a batch of PIL Images with Groq LLM."""
    
    image_content = []
    for img in pil_images:
        # Resize image if too large to save tokens
        if max(img.size) > image_max_side:
            ratio = image_max_side / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert PIL Image to base64 bytes
        import io
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG', optimize=True)
        img_bytes = img_buffer.getvalue()
        b64_image = base64.b64encode(img_bytes).decode()
        
        image_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_image}"
            }
        })
    
    expected_pages = _batch_expected_pages(start_page, len(pil_images))
    prompt = _build_transcription_prompt(expected_pages)
    max_tokens = _batch_max_tokens(expected_pages)

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        *image_content
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=0.0,
        )

        choice = response.choices[0]
        if choice.finish_reason and choice.finish_reason != "stop":
            raise GroqConversionError(
                f"Groq response finished abnormally ({choice.finish_reason}) for pages {expected_pages}"
            )

        content = choice.message.content or ""
        logger.info(
            "Raw Groq response for batch %d pages=%s: %s",
            batch_idx + 1,
            expected_pages,
            content[:500] + "...",
        )

        parsed = _decode_groq_json_response(content)
        return _normalize_batch_result(parsed)

    except Exception as e:
        raise GroqConversionError(f"Groq API request failed: {e}") from e


def _create_docx_bytes_from_content(content: dict[str, Any]) -> bytes:
    """Create DOCX bytes from Groq transcription content."""
    import io

    doc = Document()
    _write_transcription_pages_to_doc(doc, content)

    docx_buffer = io.BytesIO()
    doc.save(docx_buffer)
    docx_bytes = docx_buffer.getvalue()

    logger.info("DOCX document created in memory: %d bytes", len(docx_bytes))
    return docx_bytes


def _write_transcription_pages_to_doc(doc, content: dict[str, Any]) -> None:
    pages = content.get("pages", [])
    if not isinstance(pages, list):
        raise GroqConversionError("Groq content has invalid pages structure")

    logger.info("Writing %d page(s) to DOCX", len(pages))
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number", index + 1)
        page_text = _extract_page_text(page)
        logger.info(
            "Page %s: transcription chars=%d",
            page_number,
            _normalized_text_length(page_text),
        )

        _append_page_text(doc, page_text)
        if index < len(pages) - 1:
            doc.add_page_break()


def _append_page_text(doc, page_text: str) -> None:
    normalized_text = page_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized_text.split("\n")

    wrote_any = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        doc.add_paragraph(line)
        wrote_any = True

    if not wrote_any:
        doc.add_paragraph(" ")


def _pdf_to_images(pdf_path: Path, output_dir: Path) -> list[Path]:
    """Convert PDF pages to PNG images."""
    try:
        # Use configurable DPI for image extraction quality
        subprocess.run([
            "pdftoppm", "-png", "-r", str(settings.groq_pdf_image_dpi),
            str(pdf_path), str(output_dir / "page")
        ], check=True, capture_output=True)
        
        # Sort files numerically: page-1.png, page-2.png, ...
        images = sorted(
            output_dir.glob("page-*.png"),
            key=lambda p: int(p.stem.split('-')[-1])
        )
        return images
        
    except subprocess.CalledProcessError as e:
        raise GroqConversionError(f"Failed to convert PDF to images: {e.stderr}") from e


def _process_images_with_groq(image_paths: list[Path]) -> dict[str, Any]:
    """Process images with adaptive Groq batching strategy."""

    merged = _process_images_with_adaptive_strategy(
        image_paths,
        _process_batch_with_groq,
    )
    _validate_merged_pages(merged, expected_pages=len(image_paths))
    return merged


def _process_batch_with_groq(
    client,
    image_paths: list[Path],
    start_page: int,
    batch_idx: int,
    image_max_side: int,
) -> dict[str, Any]:
    """Process a single batch of images (up to 5) with Groq LLM."""

    image_content = []
    for img_path in image_paths:
        # Resize image if too large to save tokens (reduced resolution)
        with Image.open(img_path) as img:
            if max(img.size) > image_max_side:
                ratio = image_max_side / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save resized image
                resized_path = img_path.with_suffix('.resized.png')
                img.save(resized_path, 'PNG')
                img_path = resized_path
        
        # Encode to base64
        with open(img_path, 'rb') as f:
            b64_image = base64.b64encode(f.read()).decode()
        
        image_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_image}"
            }
        })

    expected_pages = _batch_expected_pages(start_page, len(image_paths))
    prompt = _build_transcription_prompt(expected_pages)
    max_tokens = _batch_max_tokens(expected_pages)

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *image_content
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=0.0,
        )

        choice = response.choices[0]
        if choice.finish_reason and choice.finish_reason != "stop":
            raise GroqConversionError(
                f"Groq response finished abnormally ({choice.finish_reason}) for pages {expected_pages}"
            )

        content = choice.message.content or ""
        logger.info(
            "Raw Groq response for batch %d pages=%s: %s",
            batch_idx + 1,
            expected_pages,
            content[:2000] + "...",
        )

        parsed = _decode_groq_json_response(content)
        normalized = _normalize_batch_result(parsed)
        pages = normalized.get("pages", [])
        logger.info(
            "Parsed Groq JSON: pages_count=%d page_numbers=%s",
            len(pages),
            [p.get("page_number") for p in pages],
        )
        return normalized

    except Exception as e:
        raise GroqConversionError(f"Groq API request failed: {e}") from e


def _build_initial_tasks(images: list[T]) -> deque[_GroqTask]:
    batch_size = settings.groq_batch_size
    enumerated = [(idx + 1, image) for idx, image in enumerate(images)]
    tasks: deque[_GroqTask] = deque()
    for i in range(0, len(enumerated), batch_size):
        tasks.append(
            _GroqTask(
                page_items=enumerated[i:i + batch_size],
                image_max_side=settings.groq_image_max_side,
                retries_left=settings.groq_retry_per_task,
            )
        )
    return tasks


def _split_task(task: _GroqTask) -> tuple[_GroqTask, _GroqTask] | None:
    if len(task.page_items) <= settings.groq_min_batch_size:
        return None

    midpoint = len(task.page_items) // 2
    if midpoint < settings.groq_min_batch_size:
        return None

    left_pages = task.page_items[:midpoint]
    right_pages = task.page_items[midpoint:]
    if len(left_pages) < settings.groq_min_batch_size or len(right_pages) < settings.groq_min_batch_size:
        return None

    return (
        _GroqTask(
            page_items=left_pages,
            image_max_side=task.image_max_side,
            retries_left=task.retries_left,
        ),
        _GroqTask(
            page_items=right_pages,
            image_max_side=task.image_max_side,
            retries_left=task.retries_left,
        ),
    )


def _downscale_task(task: _GroqTask) -> _GroqTask | None:
    if task.image_max_side <= settings.groq_min_image_max_side:
        return None

    reduced = int(task.image_max_side * settings.groq_image_side_reduction_factor)
    if reduced >= task.image_max_side:
        reduced = task.image_max_side - 1
    if reduced < settings.groq_min_image_max_side:
        reduced = settings.groq_min_image_max_side
    if reduced >= task.image_max_side:
        return None

    return _GroqTask(
        page_items=task.page_items,
        image_max_side=reduced,
        retries_left=task.retries_left,
    )


def _format_page_numbers(page_numbers: list[int]) -> str:
    if not page_numbers:
        return "[]"
    if len(page_numbers) == 1:
        return str(page_numbers[0])
    return f"{page_numbers[0]}-{page_numbers[-1]}"


def _validate_batch_pages(batch_result: dict[str, Any], expected_pages: list[int]) -> None:
    pages = batch_result.get("pages", [])
    actual_page_numbers: list[int] = []
    for page in pages:
        raw_page_number = page.get("page_number")
        try:
            actual_page_numbers.append(int(raw_page_number))
        except (TypeError, ValueError) as exc:
            raise GroqConversionError(
                f"Groq batch output has invalid page number: {raw_page_number!r}"
            ) from exc

    if len(actual_page_numbers) != len(set(actual_page_numbers)):
        raise GroqConversionError(f"Groq batch output has duplicate pages: {actual_page_numbers}")

    expected_set = set(expected_pages)
    actual_set = set(actual_page_numbers)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        raise GroqConversionError(
            "Groq batch output incomplete "
            f"(expected={sorted(expected_set)}, actual={sorted(actual_set)}, missing={missing}, extra={extra})"
        )

    total_chars = sum(_normalized_text_length(_extract_page_text(page)) for page in pages)
    minimum_expected_chars = max(
        _DEFAULT_MIN_TEXT_CHARS_PER_PAGE,
        len(expected_pages) * _DEFAULT_MIN_TEXT_CHARS_PER_PAGE,
    )
    if total_chars < minimum_expected_chars:
        raise GroqConversionError(
            "Groq batch output is suspiciously short "
            f"(chars={total_chars}, min_expected={minimum_expected_chars}, pages={expected_pages})"
        )


def _merge_batch_results(batch_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge results from multiple batches into single document structure."""
    
    if not batch_results:
        raise GroqConversionError("No batch results to merge")
    
    merged_title = _safe_text(batch_results[0].get("title", ""))
    best_pages: dict[int, dict[str, Any]] = {}

    for batch_result in batch_results:
        pages = batch_result.get("pages", [])
        if not isinstance(pages, list):
            continue

        for page in pages:
            if not isinstance(page, dict):
                continue
            raw_page_number = page.get("page_number")
            try:
                page_number = int(raw_page_number)
            except (TypeError, ValueError):
                continue

            candidate = dict(page)
            candidate["page_number"] = page_number
            candidate["text"] = _extract_page_text(candidate)

            existing = best_pages.get(page_number)
            if existing is None:
                best_pages[page_number] = candidate
                continue

            if _normalized_text_length(candidate["text"]) > _normalized_text_length(
                _extract_page_text(existing)
            ):
                best_pages[page_number] = candidate

    merged_pages = [best_pages[number] for number in sorted(best_pages)]
    logger.info(
        "Merged %d batches into %d unique pages",
        len(batch_results),
        len(merged_pages),
    )
    return {"title": merged_title, "pages": merged_pages}


def _validate_merged_pages(content: dict[str, Any], expected_pages: int) -> None:
    """Ensure merged Groq result contains a complete contiguous page set."""
    pages = content.get("pages", [])
    page_numbers: list[int] = []
    for page in pages:
        raw_page_number = page.get("page_number")
        if raw_page_number is None:
            continue
        try:
            page_numbers.append(int(raw_page_number))
        except (TypeError, ValueError) as exc:
            raise GroqConversionError(
                f"Groq output has invalid page number: {raw_page_number!r}"
            ) from exc

    if len(page_numbers) != len(set(page_numbers)):
        raise GroqConversionError(f"Groq output has duplicate pages: {page_numbers}")

    expected = set(range(1, expected_pages + 1))
    actual = set(page_numbers)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing or extra:
        logger.error(
            "Groq page validation failed. expected=%s actual=%s missing=%s extra=%s",
            sorted(expected),
            sorted(actual),
            missing,
            extra,
        )
        raise GroqConversionError(
            f"Groq output incomplete (missing pages: {missing}, extra pages: {extra})"
        )

    total_chars = sum(_normalized_text_length(_extract_page_text(page)) for page in pages)
    minimum_expected_chars = max(
        _DEFAULT_MIN_TEXT_CHARS_PER_PAGE,
        expected_pages * _DEFAULT_MIN_TEXT_CHARS_PER_PAGE,
    )
    if total_chars < minimum_expected_chars:
        raise GroqConversionError(
            "Groq merged output is suspiciously short "
            f"(chars={total_chars}, min_expected={minimum_expected_chars})"
        )


def _create_docx_from_content(content: dict[str, Any], output_path: Path) -> None:
    """Create DOCX document from Groq transcription content."""

    doc = Document()
    _write_transcription_pages_to_doc(doc, content)
    doc.save(output_path)
    logger.info("DOCX document saved to %s", output_path)


def _add_header_row(doc, layout: dict, formatting: dict, content: str = "") -> None:
    """Add header row with left and right aligned text.
    
    If layout.left_text and layout.right_text are both empty (common when Groq
    fills only the 'content' field), falls back to rendering 'content' as a
    heading paragraph so nothing is silently dropped.
    """
    left_text = layout.get("left_text", "")
    right_text = layout.get("right_text", "")
    
    if not left_text and not right_text:
        # Fallback: render the section content as a heading
        if content:
            font_size = formatting.get("font_size", "normal")
            level = 1 if font_size == "large" else 2
            para = doc.add_heading(content, level=level)
            _apply_paragraph_alignment(para, formatting)
            logger.debug("_add_header_row: rendered content as heading level=%d: %s", level, content[:80])
        return
        
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    # Add left text
    if left_text:
        left_run = para.add_run(left_text)
        _apply_text_formatting(left_run, formatting)
    
    # Add tab to push right text to the right
    para.add_run("\t")
    
    # Add right text
    if right_text:
        right_run = para.add_run(right_text)
        _apply_text_formatting(right_run, formatting)
    
    # Set tab stop at right margin
    tab_stops = para.paragraph_format.tab_stops
    tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)


def _add_formatted_heading(doc, text: str, formatting: dict) -> None:
    """Add heading with formatting."""
    para = doc.add_heading(text, level=2)
    
    if para.runs:
        run = para.runs[0]
        _apply_text_formatting(run, formatting)
    
    _apply_paragraph_alignment(para, formatting)


def _add_formatted_paragraph(doc, text: str, formatting: dict) -> None:
    """Add paragraph with formatting.
    
    Splits text on newlines so that multi-line content from Groq
    (e.g. bulleted job-duty lists joined by \\n) becomes individual
    DOCX paragraphs instead of one compressed block.
    """
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        para = doc.add_paragraph()
        _add_text_with_number_highlighting(para, line, formatting)
        _apply_paragraph_alignment(para, formatting)


def _add_text_with_buyrug_coloring(para, text: str, base_formatting: dict) -> None:
    """Add text with special green coloring for BUYRUG words."""
    import re
    
    # Find BUYRUG-related words and make them green
    words = re.split(r'(\s+)', text)  # Split by whitespace but keep separators
    
    for word in words:
        if 'BUYRUG' in word.upper() or 'BUYRUQ' in word.upper():
            # Make BUYRUG words green and bold
            run = para.add_run(word)
            green_formatting = base_formatting.copy()
            green_formatting["color"] = "green"
            green_formatting["bold"] = True
            _apply_text_formatting(run, green_formatting)
        elif word.isdigit():
            # Keep number highlighting
            run = para.add_run(word)
            number_formatting = base_formatting.copy()
            number_formatting["bold"] = True
            _apply_text_formatting(run, number_formatting)
        else:
            # Regular text
            run = para.add_run(word)
            _apply_text_formatting(run, base_formatting)


def _add_formatted_list(doc, text: str, formatting: dict) -> None:
    """Add formatted list items."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines:
        para = doc.add_paragraph(line, style='List Bullet')
        if para.runs:
            _apply_text_formatting(para.runs[0], formatting)


def _add_formatted_table(doc, table_data: dict, fallback_text: str) -> None:
    """Add properly formatted table with borders and optimized column widths."""
    headers = table_data.get("headers", [])
    rows = table_data.get("rows", [])
    has_borders = table_data.get("has_borders", True)
    header_styling = table_data.get("header_styling", True)
    column_types = table_data.get("column_types", [])
    
    if not headers and not rows:
        # Fallback to parsing table from text
        _add_table_from_text(doc, fallback_text)
        return
    
    # Create table
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Add headers
    header_row = table.rows[0]
    for i, header in enumerate(headers):
        cell = header_row.cells[i]
        cell.text = str(header)
        
        if header_styling:
            # Bold header text
            if cell.paragraphs and cell.paragraphs[0].runs:
                cell.paragraphs[0].runs[0].bold = True
            # Center align headers
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add data rows
    for row_data in rows:
        row = table.add_row()
        for i, cell_data in enumerate(row_data):
            if i < len(row.cells):
                row.cells[i].text = str(cell_data)
                # Right align numbers
                if str(cell_data).replace(' ', '').replace(',', '').isdigit():
                    row.cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    
    # Optimize column widths based on content
    _auto_adjust_column_widths(table, headers, column_types)
    
    # Apply table styling
    if has_borders:
        _set_table_borders(table)


def _add_table_from_text(doc, text: str) -> None:
    """Parse table from text and create formatted table with optimized widths."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    table_rows = []
    
    for line in lines:
        if '|' in line:
            # Parse pipe-separated table
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                table_rows.append(cells)
    
    if not table_rows:
        # Just add as paragraph if no table structure found
        para = doc.add_paragraph(text)
        return
    
    # Create table from parsed data
    max_cols = max(len(row) for row in table_rows)
    table = doc.add_table(rows=len(table_rows), cols=max_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    headers = table_rows[0] if table_rows else []
    
    for i, row_data in enumerate(table_rows):
        for j, cell_data in enumerate(row_data):
            if j < len(table.rows[i].cells):
                table.rows[i].cells[j].text = cell_data
                
                # Style first row as header
                if i == 0:
                    cell = table.rows[i].cells[j]
                    if cell.paragraphs and cell.paragraphs[0].runs:
                        cell.paragraphs[0].runs[0].bold = True
                    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                # Right align numbers in data rows
                elif str(cell_data).replace(' ', '').replace(',', '').replace('.', '').isdigit():
                    table.rows[i].cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    
    # Auto-adjust column widths
    _auto_adjust_column_widths(table, headers, [])
    
    _set_table_borders(table)


def _add_text_with_number_highlighting(para, text: str, base_formatting: dict) -> None:
    """Add text with numbers highlighted."""
    import re
    
    # Split text by numbers
    parts = re.split(r'(\d+)', text)
    
    for part in parts:
        if part.isdigit():
            # Highlight numbers
            run = para.add_run(part)
            number_formatting = base_formatting.copy()
            number_formatting["bold"] = True
            _apply_text_formatting(run, number_formatting)
        else:
            # Regular text
            run = para.add_run(part)
            _apply_text_formatting(run, base_formatting)


def _apply_text_formatting(run, formatting: dict) -> None:
    """Apply formatting to a text run."""
    if formatting.get("bold"):
        run.bold = True
    
    if formatting.get("italic"):
        run.italic = True
    
    # Apply color - expanded color support
    color = formatting.get("color", "black")
    if color == "green":
        run.font.color.rgb = RGBColor(0, 128, 0)  # Standard green
    elif color == "blue":
        run.font.color.rgb = RGBColor(0, 0, 255)
    elif color == "red":
        run.font.color.rgb = RGBColor(255, 0, 0)
    elif color == "dark_blue":
        run.font.color.rgb = RGBColor(0, 0, 139)
    elif color == "orange":
        run.font.color.rgb = RGBColor(255, 165, 0)
    # Default is black, no need to set explicitly
    
    # Apply font size
    font_size = formatting.get("font_size", "normal")
    if font_size == "large":
        run.font.size = Pt(14)
    elif font_size == "small":
        run.font.size = Pt(10)
    else:
        run.font.size = Pt(12)


def _auto_adjust_column_widths(table, headers: list, column_types: list) -> None:
    """Automatically adjust column widths based on content type."""
    try:
        for i, header in enumerate(headers):
            header_lower = str(header).lower()
            
            # Determine column type
            if i < len(column_types):
                col_type = column_types[i]
            else:
                # Auto-detect from header name
                if header_lower in ['№', '#', 'no', 'num']:
                    col_type = "number"
                elif 'nomi' in header_lower or 'название' in header_lower or 'name' in header_lower:
                    col_type = "text"
                else:
                    col_type = "number"  # Default for numeric columns
            
            # Set width based on type
            if col_type == "number" and header_lower in ['№', '#', 'no', 'soni']:
                # Very narrow for row numbers and quantities
                table.columns[i].width = Inches(0.6)
            elif col_type == "text" or 'nomi' in header_lower:
                # Wide for text descriptions
                table.columns[i].width = Inches(3.2)
            elif header_lower in ['narxi', 'jami', 'price', 'total', 'сумма']:
                # Medium width for money amounts
                table.columns[i].width = Inches(1.4)
            else:
                # Default medium width
                table.columns[i].width = Inches(1.2)
                
    except Exception as e:
        logger.warning("Could not adjust column widths: %s", e)
        # Continue without width adjustment


def _apply_paragraph_alignment(para, formatting: dict) -> None:
    """Apply paragraph alignment."""
    alignment = formatting.get("alignment", "left")
    if alignment == "center":
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif alignment == "right":
        para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    elif alignment == "justified":
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def _set_table_borders(table) -> None:
    """Set borders for table."""
    try:
        # This is a simplified border setting - full implementation would need more XML manipulation
        for row in table.rows:
            for cell in row.cells:
                # Set cell borders through XML if needed
                pass
    except Exception as e:
        logger.warning("Could not set table borders: %s", e)
