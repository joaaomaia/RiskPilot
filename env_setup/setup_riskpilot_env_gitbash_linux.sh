#!/usr/bin/env bash
# ------------------------------------------------------------------
# setup_riskpilot_env.sh  (Linux version, Git Bash or native bash)
# ------------------------------------------------------------------
# 1. Remove previous venv (if any)
# 2. Create fresh venv with Python 3.11 (fallback to python3)
# 3. Activate venv
# 4. Install pip 23.3.2 + pip‑tools 7.3.0 (compatible)
# 5. Generate requirements.lock with extras viz & binning
# ------------------------------------------------------------------

set -e

ENV_PATH="$HOME/envs/riskpilot-lock"
PROJECT_DIR="${PROJECT_DIR:-$HOME/RiskPilot}"

# Detect Python executable
PYTHON_EXEC=$(command -v python3.11 || command -v python3 || true)
if [[ -z "$PYTHON_EXEC" ]]; then
  echo "❌ Python 3 not found. Install Python 3.11 or 3.10 first."
  exit 1
fi

echo "🧹 Removing previous env (if any)..."
rm -rf "$ENV_PATH"

echo "🐍 Creating new virtualenv at $ENV_PATH using $PYTHON_EXEC ..."
"$PYTHON_EXEC" -m venv "$ENV_PATH"

echo "✅ Activating environment..."
source "$ENV_PATH/bin/activate"

echo "🚀 Upgrading build tools (pip, setuptools, wheel)..."
python -m pip install --upgrade pip setuptools wheel

echo "📦 Installing pip 23.3.2 + pip‑tools 7.3.0 ..."
python -m pip install pip==23.3.2 pip-tools==7.3.0

echo "📂 Entering project directory: $PROJECT_DIR"
cd "$PROJECT_DIR" || { echo "❌ Project directory not found"; exit 1; }


# não precisa criar requirements.lock em outras maquinas, só na máquina de desenvolvimento

# echo "🔒 Generating requirements.lock (extras: viz, binning)..."
# pip-compile --extra viz --extra binning --generate-hashes --output-file requirements.lock pyproject.toml

# echo "✅ Environment 'riskpilot-lock' ready and requirements.lock generated!"
