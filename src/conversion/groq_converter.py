"""Groq LLM-based PDF to DOCX conversion with vision models."""

from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    import groq
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from PIL import Image
except ImportError:
    groq = None
    Document = None

from ..config.settings import settings

logger = logging.getLogger(__name__)


class GroqConversionError(RuntimeError):
    """Signals that Groq LLM conversion failed."""


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


def _pdf_to_images(pdf_path: Path, output_dir: Path) -> list[Path]:
    """Convert PDF pages to PNG images."""
    try:
        # Use higher DPI for better OCR quality
        subprocess.run([
            "pdftoppm", "-png", "-r", "200", 
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
    """Process images with Groq vision LLM in batches (max 5 images per request)."""
    
    client = groq.Groq(api_key=settings.groq_api_key)
    
    # Split into batches of 3 images (reduced from 5 for better quality)
    batches = [image_paths[i:i+3] for i in range(0, len(image_paths), 3)]
    logger.info("Processing %d images in %d batches", len(image_paths), len(batches))
    
    all_results = []
    
    for batch_idx, batch_images in enumerate(batches):
        logger.info("Processing batch %d/%d with %d images", batch_idx + 1, len(batches), len(batch_images))
        
        try:
            batch_result = _process_batch_with_groq(client, batch_images, batch_idx)
            all_results.append(batch_result)
        except Exception as e:
            logger.error("Failed to process batch %d: %s", batch_idx + 1, e)
            # Continue with other batches instead of failing completely
            continue
    
    if not all_results:
        raise GroqConversionError("All batches failed to process")
    
    # Merge all batch results into single document
    return _merge_batch_results(all_results)


def _process_batch_with_groq(client, image_paths: list[Path], batch_idx: int) -> dict[str, Any]:
    """Process a single batch of images (up to 5) with Groq LLM."""
    
    # Calculate actual page numbers for this batch (3 images per batch)
    start_page = batch_idx * 3 + 1
    page_numbers = list(range(start_page, start_page + len(image_paths)))
    image_content = []
    for img_path in image_paths:
        # Resize image if too large to save tokens (reduced resolution)
        with Image.open(img_path) as img:
            # Resize to max 800px on longest side (reduced from 1024)
            if max(img.size) > 800:
                ratio = 800 / max(img.size) 
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

    # Optimized prompt for concise but complete extraction
    prompt = f"""Extract ALL text from these {len(image_paths)} pages (pages {start_page}-{start_page + len(image_paths) - 1}). Be COMPLETE but CONCISE.

    JSON format:
    {{
        "title": "doc title (only batch 1)",
        "pages": [
            {{
                "page_number": {start_page},
                "sections": [
                    {{
                        "type": "heading|paragraph|table|list",
                        "content": "FULL TEXT - do not truncate",
                        "formatting": {{"bold": true/false, "alignment": "left|center"}}
                    }}
                ]
            }}
        ]
    }}
    
    CRITICAL:
    - Extract EVERY word, number, table row
    - For tables: use "| col1 | col2 |" format with ALL rows
    - Page numbers: {start_page} to {start_page + len(image_paths) - 1}
    - Russian/Uzbek/English text
    - No summaries - extract everything verbatim
    """

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
            max_tokens=8000,  # Maximum allowed for this model
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        logger.info("Raw Groq response for batch %d: %s", batch_idx + 1, content[:2000] + "...")
        
        import json
        result = json.loads(content)
        pages = result.get('pages', [])
        logger.info("Parsed JSON structure: title=%s, pages_count=%d, page_numbers=%s", 
                   type(result.get('title')), len(pages), 
                   [p.get('page_number') for p in pages])
        return result
        
    except Exception as e:
        raise GroqConversionError(f"Groq API request failed: {e}") from e



def _merge_batch_results(batch_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge results from multiple batches into single document structure."""
    
    if not batch_results:
        raise GroqConversionError("No batch results to merge")
    
    # Use title from first batch
    merged = {
        "title": batch_results[0].get("title", ""),
        "pages": []
    }
    
    # Combine all pages from all batches
    for batch_result in batch_results:
        pages = batch_result.get("pages", [])
        merged["pages"].extend(pages)
    
    # Sort pages by page number to ensure correct order
    merged["pages"].sort(key=lambda p: p.get("page_number", 0))
    
    logger.info("Merged %d batches into %d total pages", len(batch_results), len(merged["pages"]))
    return merged


def _create_docx_from_content(content: dict[str, Any], output_path: Path) -> None:
    """Create DOCX document from structured content."""
    
    def safe_text(value: Any) -> str:
        """Safely convert any value to string, handling lists."""
        if isinstance(value, list):
            return ' '.join(str(item) for item in value if item)
        return str(value) if value else ""
    
    doc = Document()
    
    # Add title if present
    title = content.get("title")
    if title:
        title_text = safe_text(title).strip()
        if title_text:
            title_para = doc.add_heading(title_text, level=1)
            title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            logger.info("Added document title: %s", title_text[:50])
    
    # Process each page
    pages = content.get("pages", [])
    logger.info("Processing %d pages for DOCX creation", len(pages))
    
    for page_idx, page in enumerate(pages):
        page_num = page.get("page_number", page_idx + 1)
        sections = page.get("sections", [])
        logger.info("Page %d: processing %d sections", page_num, len(sections))
        
        for section_idx, section in enumerate(sections):
            section_type = section.get("type", "paragraph")
            text_content = safe_text(section.get("content", "")).strip()
            formatting = section.get("formatting", {})
            
            if not text_content:
                logger.debug("Skipping empty section %d on page %d", section_idx, page_num)
                continue
            
            logger.debug("Adding %s: %s", section_type, text_content[:100])
                
            if section_type == "heading":
                para = doc.add_heading(text_content, level=2)
            elif section_type == "list":
                # Split by lines and create list items
                lines = [line.strip() for line in text_content.split('\n') if line.strip()]
                for line in lines:
                    para = doc.add_paragraph(line, style='List Bullet')
            else:  # paragraph or table (simple paragraph for now)
                para = doc.add_paragraph(text_content)
            
            # Apply formatting to last added paragraph
            if hasattr(para, 'runs') and para.runs:
                run = para.runs[0]
                if formatting.get("bold"):
                    run.bold = True
                if formatting.get("italic"):
                    run.italic = True
            
            # Apply alignment
            alignment = formatting.get("alignment", "left")
            if alignment == "center":
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif alignment == "right":
                para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    
    # Save document
    doc.save(output_path)
    logger.info("DOCX document saved to %s", output_path)