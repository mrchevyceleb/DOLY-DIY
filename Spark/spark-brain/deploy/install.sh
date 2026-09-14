#!/usr/bin/env bash
# Spark brain — installer for Doly's Raspberry Pi.
# Run as root ON THE ROBOT:  sudo bash install.sh
set -euo pipefail

APP_DIR=/opt/spark/app
VENV=/opt/spark/venv
VOSK_DIR=/opt/spark/vosk-model-small-en-us-0.15
VOSK_URL="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.tar.gz"

[[ $EUID -eq 0 ]] || { echo "Run with sudo (on the robot)."; exit 1; }
command -v python3 >/dev/null || { echo "python3 required"; exit 1; }

echo "==> Checking Doly SDK modules…"
if ! python3 -c "import doly_tts, doly_sound, doly_eye, doly_touch" 2>/dev/null; then
  echo "WARNING: doly_* modules not found in system python."
  echo "         Voice/eyes/touch may not work — check the Doly image."
fi

echo "==> Installing system packages…"
apt-get update -qq
apt-get install -y -qq python3-venv python3-dev alsa-utils >/dev/null

echo "==> Deploying app to ${APP_DIR}…"
mkdir -p /opt/spark/state /opt/spark/vosk "${APP_DIR}"
rsync -a --delete spark/ "${APP_DIR}/spark/" 2>/dev/null || { rm -rf "${APP_DIR}/spark"; cp -r spark/ "${APP_DIR}/"; }
cp -f prompt.md config.json requirements.txt "${APP_DIR}/"

if [ ! -d "${VENV}" ]; then
  echo "==> Creating venv (system-site-packages so doly_* stays visible)…"
  python3 -m venv --system-site-packages "${VENV}"
fi

echo "==> Installing python deps…"
"${VENV}/bin/pip" install --quiet --upgrade pip
"${VENV}/bin/pip" install --quiet -r "${APP_DIR}/requirements.txt"

if [ ! -d "${VOSK_DIR}" ]; then
  echo "==> Downloading Vosk ASR model (~40 MB)…"
  curl -fsSL "${VOSK_URL}" -o /tmp/vosk.tar.gz
  tar -xzf /tmp/vosk.tar.gz -C /opt/spark/
  rm -f /tmp/vosk.tar.gz
fi

echo "==> Installing spark CLI…"
cat > /usr/local/bin/spark <<'SPARKEOF'
#!/usr/bin/env bash
set -uo pipefail
# self-elevate: systemctl needs root; doly user has passwordless sudo
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if sudo -n true 2>/dev/null; then SUDO="sudo"; else echo "Need root (try: sudo spark $*)"; exit 1; fi
fi
systemctl() { command $SUDO systemctl "$@"; }
case "${1:-}" in
  on)
    restore_doly() {
      systemctl disable --now spark-brain 2>/dev/null
      systemctl enable --now doly 2>/dev/null
      if systemctl is-active --quiet doly; then
        echo "Stock doly restored."
      else
        echo "WARNING: doly failed to restart — check: systemctl status doly"
      fi
      journalctl -u spark-brain -n 20 --no-pager
    }
    # Type=notify: enable --now blocks until READY=1 (real init) or timeout
    if ! systemctl enable --now spark-brain; then
      echo "spark-brain failed to initialize — rolling back."
      restore_doly
      exit 1
    fi
    if ! timeout 90 bash -c 'until systemctl is-active --quiet spark-brain; do sleep 1; done'; then
      echo "spark-brain did not reach ready — rolling back."
      restore_doly
      exit 1
    fi
    # stability check: catch instant crash-loops before parking stock doly
    sleep 5
    if ! systemctl is-active --quiet spark-brain; then
      echo "spark-brain crashed right after startup — rolling back."
      restore_doly
      exit 1
    fi
    systemctl disable --now doly 2>/dev/null || true
    echo "Spark online. Stock doly parked (spark off to restore)."
    ;;
  off)
    systemctl disable --now spark-brain
    if systemctl enable --now doly 2>/dev/null && systemctl is-active --quiet doly; then
      echo "Stock Doly restored."
    else
      echo "WARNING: doly did not re-activate — check: systemctl status doly"
      exit 1
    fi
    ;;
  status) systemctl status spark-brain --no-pager -l ;;
  log)    journalctl -u spark-brain -f -n 50 ;;
  test)
    if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
      exec sudo -u "${SUDO_USER}" /opt/spark/venv/bin/python -m spark --text
    else
      exec /opt/spark/venv/bin/python -m spark --text
    fi ;;
  *) echo "usage: spark {on|off|status|log|test}" ;;
esac
SPARKEOF

chmod +x /usr/local/bin/spark

echo "==> Installing systemd unit…"
cp deploy/spark-brain.service /etc/systemd/system/
systemctl daemon-reload

echo
echo "DONE. On the robot run:"
echo "  spark test    # text REPL to verify the brain pipeline"
echo "  spark on      # go live (voice mode)"
echo "  journalctl -u spark-brain -f   # watch it think"
