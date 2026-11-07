"""Состояние, которое меняется без правки .env: группы, лимит и счётчики за сутки.

Хранится в JSON-файле (STATE_FILE) и переживает перезапуск и деплой.
Все обработчики работают в одном потоке asyncio, поэтому блокировки не нужны.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class GroupInfo:
    title: str = ""
    added_by: int | None = None
    added_at: str = ""


class BotState:
    """Группы, подключённые супер-админом, убранные группы, лимит и счётчики."""

    def __init__(self, path: Path, timezone: str = "UTC") -> None:
        self._path = Path(path)
        self._zone = _zone(timezone)
        self._groups: dict[int, GroupInfo] = {}
        self._titles: dict[int, str] = {}
        self._removed: set[int] = set()
        self._limit: int | None = None
        self._day = ""
        self._usage: dict[int, int] = {}
        self._load()

    # --- группы ---------------------------------------------------------

    def is_group_allowed(self, chat_id: int) -> bool:
        if chat_id in self._removed:
            return False
        if not settings.allowed_users_only:
            return True
        return chat_id in settings.allowed_chat_ids or chat_id in self._groups

    def allowed_groups(self) -> dict[int, str]:
        """Группы с доступом и их названия (из .env и подключённые супер-админом)."""
        ids = (set(settings.allowed_chat_ids) | set(self._groups)) - self._removed
        return {chat_id: self.title(chat_id) for chat_id in sorted(ids)}

    def title(self, chat_id: int) -> str:
        info = self._groups.get(chat_id)
        return (info.title if info else "") or self._titles.get(chat_id, "")

    def activate_group(self, chat_id: int, title: str | None, by: int) -> None:
        self._removed.discard(chat_id)
        if chat_id not in self._groups:
            self._groups[chat_id] = GroupInfo(title or "", by, _now(self._zone).isoformat(timespec="seconds"))
        self._save()

    def remove_group(self, chat_id: int) -> None:
        """До следующего вызова бота супер-админом в этой группе."""
        self._groups.pop(chat_id, None)
        self._removed.add(chat_id)
        self._save()

    def migrate_group(self, old_id: int, new_id: int) -> bool:
        """Группа стала супергруппой: доступ переходит на новый ID. True — был доступ."""
        if not self.is_group_allowed(old_id):
            return False
        info = self._groups.pop(old_id, None) or GroupInfo(self.title(old_id))
        self._removed.discard(new_id)
        self._groups[new_id] = info
        self._save()
        return True

    def remember_title(self, chat_id: int, title: str | None) -> None:
        if not title or self.title(chat_id) == title:
            return
        if chat_id in self._groups:
            self._groups[chat_id].title = title
        else:
            self._titles[chat_id] = title
        self._save()

    # --- лимит ----------------------------------------------------------

    @property
    def daily_limit(self) -> int:
        """Файлов в сутки на участника; 0 — без лимита."""
        return settings.daily_file_limit if self._limit is None else self._limit

    def set_daily_limit(self, limit: int) -> None:
        self._limit = max(0, int(limit))
        self._save()

    def try_consume(self, user_id: int) -> bool:
        """Забирает один файл из лимита участника; False — лимит исчерпан."""
        self._roll_day()
        limit = self.daily_limit
        used = self._usage.get(user_id, 0)
        if limit and used >= limit:
            return False
        self._usage[user_id] = used + 1
        self._save()
        return True

    def refund(self, user_id: int) -> None:
        """Возвращает файл в лимит, если конвертация не удалась."""
        self._roll_day()
        if self._usage.get(user_id, 0) > 0:
            self._usage[user_id] -= 1
            if not self._usage[user_id]:
                del self._usage[user_id]
            self._save()

    def used(self, user_id: int) -> int:
        self._roll_day()
        return self._usage.get(user_id, 0)

    def usage_today(self) -> tuple[int, int]:
        """(файлов за сегодня, участников)."""
        self._roll_day()
        return sum(self._usage.values()), len(self._usage)

    def reset_usage(self) -> None:
        self._usage.clear()
        self._save()

    @property
    def zone_name(self) -> str:
        return str(self._zone)

    # --- файл -----------------------------------------------------------

    def _roll_day(self) -> None:
        today = _now(self._zone).date().isoformat()
        if today != self._day:
            self._day = today
            if self._usage:
                self._usage.clear()
                self._save()

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            logger.error("Cannot read state file %s: %s — starting empty", self._path, exc)
            return
        self._groups = {int(k): GroupInfo(**v) for k, v in raw.get("groups", {}).items()}
        self._titles = {int(k): v for k, v in raw.get("titles", {}).items()}
        self._removed = {int(v) for v in raw.get("removed", [])}
        self._limit = raw.get("daily_limit")
        usage = raw.get("usage", {})
        self._day = usage.get("date", "")
        self._usage = {int(k): int(v) for k, v in usage.get("counts", {}).items()}

    def _save(self) -> None:
        data = {
            "groups": {str(k): asdict(v) for k, v in self._groups.items()},
            "titles": {str(k): v for k, v in self._titles.items()},
            "removed": sorted(self._removed),
            "daily_limit": self._limit,
            "usage": {"date": self._day, "counts": {str(k): v for k, v in self._usage.items()}},
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError as exc:
            logger.error("Cannot save state file %s: %s", self._path, exc)


def _zone(name: str) -> datetime.tzinfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning("Unknown TIMEZONE %r, using UTC", name)
        return datetime.timezone.utc


def _now(zone: datetime.tzinfo) -> datetime.datetime:
    return datetime.datetime.now(zone)


_state: BotState | None = None


def bot_state() -> BotState:
    global _state
    if _state is None:
        _state = BotState(settings.state_file, settings.timezone)
    return _state


def use_state(state: BotState) -> None:
    """Подменить состояние (для тестов)."""
    global _state
    _state = state
