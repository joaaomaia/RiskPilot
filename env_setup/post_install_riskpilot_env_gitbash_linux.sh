#!/usr/bin/env bash
# ------------------------------------------------------------------
# post_install_riskpilot_env_gitbash_linux.sh
# Target: Git Bash / POSIX shell on Linux
# - Activates riskpilot-lock venv
# - Installs packages from env_setup/requirements.lock (includes kaleido)
# - Installs RiskPilot in editable mode with extras
# - Smoke‑tests import
# ------------------------------------------------------------------
set -e

ENV_PATH="$HOME/envs/riskpilot-lock"
PROJECT_DIR="${PROJECT_DIR:-$HOME/RiskPilot}"

# Activate env if not already
if [[ -z "$VIRTUAL_ENV" || "$VIRTUAL_ENV" != "$ENV_PATH" ]]; then
  echo "✅ Activating environment riskpilot-lock ..."
  source "$ENV_PATH/bin/activate"
fi

echo "📥 Installing dependencies from requirements.lock (with hashes) ..."
pip install --require-hashes -r "$PROJECT_DIR/env_setup/requirements.lock"

echo "📦 Installing RiskPilot (editable) + extras ..."
pip install -e "$PROJECT_DIR"[dev,viz,binning,model]
#pip install -e "$PROJECT_DIR"[dev,viz,binning, model]

echo "🔍 Smoke-test: import riskpilot"
python - <<'PY'
import riskpilot, sys
print("RiskPilot import OK – version:", getattr(riskpilot, "__version__", "unknown"))
PY

echo "🚀 Post‑setup completed. Happy coding!"
