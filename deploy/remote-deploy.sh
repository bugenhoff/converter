#!/usr/bin/env bash
# Выполняется на сервере из GitHub Actions: код уже распакован в APP_DIR.
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/converter}"
SERVICE_NAME="${SERVICE_NAME:-converter-bot}"
cd "$APP_DIR"

if [ ! -f .env ]; then
  echo "::error::Нет файла $APP_DIR/.env — создайте его по образцу .env.example"
  exit 1
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "Устанавливаю uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv sync --frozen --no-dev

for tool in pdftoppm pdfinfo pdftotext; do
  command -v "$tool" >/dev/null 2>&1 || echo "::warning::Нет $tool — установите poppler-utils"
done
command -v soffice >/dev/null 2>&1 || command -v libreoffice >/dev/null 2>&1 \
  || echo "::warning::Нет LibreOffice — .doc/.wps и DOCX → PDF работать не будут"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo -n"
fi

# DOCX/DOC/WPS → PDF рисуется шрифтами сервера (кроме встроенных в сам DOCX;
# в .doc и .wps шрифты не встраиваются). Недостающие пакеты ставятся сами,
# если есть apt и sudo без пароля.
FONT_PACKAGES=(
  "Liberation Serif|fonts-liberation"   # метрическая замена Times New Roman и Arial
  "Carlito|fonts-crosextra-carlito"     # замена Calibri
  "Caladea|fonts-crosextra-caladea"     # замена Cambria
  "Inter|fonts-inter"
  "Roboto|fonts-roboto"
  "Open Sans|fonts-open-sans"
  "Montserrat|fonts-montserrat"
  "PT Sans|fonts-paratype"
)
families="$(fc-list : family 2>/dev/null || true)"
missing=()
for entry in "${FONT_PACKAGES[@]}"; do
  grep -qiF "${entry%%|*}" <<<"$families" || missing+=("${entry##*|}")
done
if [ "${#missing[@]}" -gt 0 ]; then
  if command -v apt-get >/dev/null 2>&1 && $SUDO true 2>/dev/null; then
    echo "Устанавливаю шрифты: ${missing[*]}"
    install_fonts() { $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "$@" >/dev/null 2>&1; }
    if ! install_fonts "${missing[@]}"; then
      # Устаревший список пакетов или одного пакета нет в дистрибутиве — ставим по одному.
      $SUDO apt-get update -qq >/dev/null 2>&1 || true
      for package in "${missing[@]}"; do
        install_fonts "$package" || echo "::warning::Не удалось установить $package"
      done
    fi
  else
    echo "::warning::Нет шрифтов (${missing[*]}) — PDF будут со шрифтами-заменами. Установите: sudo apt install ${missing[*]}"
  fi
fi

sed -e "s|@APP_DIR@|$APP_DIR|g" -e "s|@USER@|$(id -un)|g" -e "s|@HOME@|$HOME|g" \
  deploy/converter-bot.service | $SUDO tee "/etc/systemd/system/$SERVICE_NAME.service" >/dev/null
$SUDO systemctl daemon-reload
$SUDO systemctl enable "$SERVICE_NAME" >/dev/null
$SUDO systemctl restart "$SERVICE_NAME"

sleep 5
if ! $SUDO systemctl is-active --quiet "$SERVICE_NAME"; then
  echo "::error::Сервис $SERVICE_NAME не запустился"
  $SUDO journalctl -u "$SERVICE_NAME" -n 50 --no-pager || true
  exit 1
fi
$SUDO journalctl -u "$SERVICE_NAME" -n 15 --no-pager || true

# Второй экземпляр с тем же токеном (запущенный вручную) ломает обоим getUpdates.
main_pid="$(systemctl show -p MainPID --value "$SERVICE_NAME")"
others="$(pgrep -f 'python.* bot\.py' | grep -vx "$main_pid" || true)"
if [ -n "$others" ]; then
  echo "::warning::Кроме сервиса работают другие процессы bot.py (PID: $(echo $others)). Остановите их, иначе Telegram отдаёт ошибку Conflict."
fi
echo "Деплой завершён: $SERVICE_NAME работает из $APP_DIR"
