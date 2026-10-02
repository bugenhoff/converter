"""Распознавание страниц vision-моделью через OpenAI-совместимый API (Groq, OpenRouter).

Каждая страница — отдельный запрос, запросы идут параллельно через общий пул
потоков: время документа близко ко времени самой долгой страницы, а не к сумме.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable

import openai

from ..config.settings import settings
from .errors import ConversionError
from .layout import Block, LayoutError, parse_page_layout
from .pages import PageImage

logger = logging.getLogger(__name__)

_MAX_TOKENS_CEILING = 131072

PROMPT = """You transcribe one scanned document page into JSON used to rebuild it as an editable Word document.

Return ONLY a JSON object {"blocks": [...]} with the page content in reading order. Omit keys whose value is the default.

Block types:
1. Paragraph: {"text": "...", "align": "justify", "first_line": true, "indent": 0.5, "size": "large", "bold": true, "gap": true}
   - text: the whole paragraph. Lines that wrap inside one paragraph are joined into one string with spaces; a word hyphenated at a line end is joined back. Use "\\n" only where the author broke a line on purpose (lines of a centered title, an address, a heading block).
   - align: "left" (default), "center", "right" or "justify" (body text whose lines reach both margins).
   - first_line: true if the first line is indented.
   - indent: left offset of the whole block as a fraction of the text width, e.g. 0.5 for a block that starts in the middle of the page. Default 0.
   - size: "small", "large" or "xlarge" relative to the main body text. Default normal.
   - bold: true if the whole block is bold. Mark bold, italic or underlined fragments inside text as **bold**, *italic*, __underlined__.
   - gap: true if there is clearly more vertical space before the block than between ordinary paragraphs.
2. Row of columns on one line, e.g. a date on the left and a number on the right, or a position on the left and a name on the right: {"row": ["left", "right"]}. 2 or 3 items, "\\n" inside an item for several lines. Optional "size", "bold", "gap".
3. Table: {"table": [["cell", "cell"], ...], "borders": false, "widths": [0.1, 0.6, 0.3]}
   - borders: false if the table has no visible ruling lines (for example a list of names and positions aligned in columns). Default true.
   - widths: approximate column widths as fractions of the text width.

Rules:
- Transcribe every word, number and punctuation mark exactly as printed, in the original language and alphabet, including apostrophes (ʻ ’ ‘) and quotes (« » “ ”). Do not translate, correct, summarize or skip anything.
- Do not transcribe watermarks or repeated diagonal background text, text inside round seals and stamps, handwritten signatures, QR codes, emblems, or page numbers.
- Return {"blocks": []} for a blank page.
- Output JSON only: no Markdown, no code fences, no comments."""


class LLMError(ConversionError):
    """Не удалось распознать страницу или документ."""


@dataclass
class PageResult:
    page: PageImage
    blocks: list[Block] | None = None
    error: str | None = None


ProgressCallback = Callable[[int, int], None]


def transcribe_pages(
    pages: Iterable[PageImage],
    total: int,
    on_progress: ProgressCallback | None = None,
) -> list[PageResult]:
    """Распознаёт страницы параллельно; ошибка страницы не роняет весь документ."""
    if not settings.llm_api_key:
        raise LLMError(f"{settings.llm_api_key_name} не задан")

    done = 0
    done_lock = threading.Lock()

    def page_finished(_: Future) -> None:
        nonlocal done
        with done_lock:
            done += 1
            current = done
        if on_progress:
            on_progress(current, total)

    futures: list[tuple[PageImage, Future]] = []
    for page in pages:
        future = _executor().submit(transcribe_page, page)
        future.add_done_callback(page_finished)
        futures.append((page, future))

    results = []
    for page, future in futures:
        try:
            results.append(PageResult(page=page, blocks=future.result()))
        except Exception as exc:
            logger.error("Page %d failed: %s", page.number, exc)
            results.append(PageResult(page=page, error=str(exc)))
    return results


def transcribe_page(page: PageImage) -> list[Block]:
    max_tokens = settings.llm_max_tokens_per_page
    attempts = settings.llm_retries + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            content, finish_reason = _request(page, max_tokens)
            if finish_reason == "length":
                # Ответ оборвался: модели с рассуждениями тратят бюджет и на них.
                max_tokens = min(max_tokens * 2, _MAX_TOKENS_CEILING)
                raise LLMError(f"ответ обрезан по max_tokens, следующая попытка с {max_tokens}")
            blocks = parse_page_layout(content)
            if not blocks and attempt < attempts:
                raise LLMError("пустой ответ")
            return blocks
        except (LLMError, LayoutError, openai.APIError) as exc:
            last_error = exc
            logger.warning("Page %d attempt %d/%d failed: %s", page.number, attempt, attempts, exc)
            if attempt < attempts and not isinstance(exc, LLMError):
                time.sleep(attempt)
    raise LLMError(f"страница {page.number}: {last_error}")


def _request(page: PageImage, max_tokens: int) -> tuple[str, str | None]:
    image_url = "data:image/jpeg;base64," + base64.b64encode(page.jpeg).decode()
    started = time.monotonic()
    response = _client().chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        temperature=0,
        **_request_options(),
    )
    elapsed = time.monotonic() - started
    _log_usage(page.number, response, elapsed)
    if not response.choices:
        raise LLMError("ответ без choices")
    choice = response.choices[0]
    return choice.message.content or "", choice.finish_reason


def _log_usage(page_number: int, response: Any, elapsed: float) -> None:
    # По этим строкам видно, куда уходит время: если токенов мало, а секунд
    # много — медленный провайдер; если токенов тысячи — объём ответа.
    usage = getattr(response, "usage", None)
    completion = getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) or 0
    logger.info(
        "LLM page %d: %.1fs, prompt=%s completion=%s (reasoning=%s), %.0f tok/s, model=%s provider=%s",
        page_number,
        elapsed,
        getattr(usage, "prompt_tokens", "?"),
        completion,
        reasoning,
        completion / elapsed if elapsed > 0 else 0,
        getattr(response, "model", "?"),
        getattr(response, "provider", "-"),
    )


def _request_options() -> dict[str, Any]:
    """Параметры, которые есть только у OpenRouter."""
    if settings.llm_provider != "openrouter":
        return {}
    extra: dict[str, Any] = {}
    if settings.openrouter_reasoning_effort:
        extra["reasoning"] = {"effort": settings.openrouter_reasoning_effort}
    if settings.openrouter_provider_sort:
        extra["provider"] = {"sort": settings.openrouter_provider_sort}
    return {"extra_body": extra} if extra else {}


@lru_cache(maxsize=1)
def _client() -> openai.OpenAI:
    # Один клиент на процесс: соединения переиспользуются между страницами.
    headers = {"X-Title": "Document Converter bot"} if settings.llm_provider == "openrouter" else None
    return openai.OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=3,
        default_headers=headers,
    )


@lru_cache(maxsize=1)
def _executor() -> ThreadPoolExecutor:
    # Общий на все документы пул: LLM_CONCURRENCY ограничивает число
    # одновременных запросов к провайдеру по всему боту.
    return ThreadPoolExecutor(max_workers=settings.llm_concurrency, thread_name_prefix="llm")
