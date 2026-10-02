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
  || echo "::warning::Нет LibreOffice — .doc конвертироваться не будут"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo -n"
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
