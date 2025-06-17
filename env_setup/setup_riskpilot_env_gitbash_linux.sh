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


#!/usr/bin/env bash
set -euo pipefail                     # aborta no 1º erro, variável não-set e pipes

ENV_PATH="$HOME/envs/riskpilot-lock"
PROJECT_DIR="${PROJECT_DIR:-$HOME/RiskPilot}"

# ──────────────────────────────────────────────────────────────
# 1. Detecta executável Python (3.11 → 3.10 → 3.9)
# ──────────────────────────────────────────────────────────────
for candidate in python3.11 python3.10 python3.9 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_EXEC=$(command -v "$candidate")
        break
    fi
done

if [[ -z "${PYTHON_EXEC:-}" ]]; then
    echo "❌ Python ≥ 3.9 não encontrado. Instale 3.11/3.10/3.9 primeiro."
    exit 1
fi

# Checa versão numérica
PY_VERSION=$("$PYTHON_EXEC" - <<'PY'
import sys, json
print(json.dumps({"major": sys.version_info.major,
                  "minor": sys.version_info.minor}))
PY
)
MAJOR=$(echo "$PY_VERSION" | jq -r .major)
MINOR=$(echo "$PY_VERSION" | jq -r .minor)
if (( MAJOR < 3 || (MAJOR == 3 && MINOR < 9) )); then
    echo "❌ Python ${MAJOR}.${MINOR} é muito antigo. Use >= 3.9."
    exit 1
fi

echo "🐍 Usando Python ${MAJOR}.${MINOR} em $PYTHON_EXEC"

# ──────────────────────────────────────────────────────────────
# 2. Cria (ou recria) o venv
# ──────────────────────────────────────────────────────────────
echo "🧹 Removendo env prévio (se existir)…"
rm -rf "$ENV_PATH"

echo "🔧 Criando virtualenv em $ENV_PATH …"
"$PYTHON_EXEC" -m venv "$ENV_PATH"

echo "✅ Ativando ambiente…"
# shellcheck source=/dev/null
source "$ENV_PATH/bin/activate"

# ──────────────────────────────────────────────────────────────
# 3. Atualiza ferramentas de build
# ──────────────────────────────────────────────────────────────
echo "🚀 Atualizando pip / setuptools / wheel…"
python -m pip install --upgrade pip setuptools wheel >/dev/null

# Pin pip 23.3.2 apenas se houver distribuição para esta versão de Python
if python -m pip install --quiet --dry-run pip==23.3.2 2>/dev/null; then
    echo "📦 Instalando pip 23.3.2 + pip-tools 7.3.0…"
    python -m pip install pip==23.3.2 pip-tools==7.3.0
else
    echo "⚠️  Wheel do pip 23.3.2 indisponível para Python ${MAJOR}.${MINOR}; mantendo versão recém-atualizada."
    python -m pip install pip-tools==7.3.0
fi

# # ──────────────────────────────────────────────────────────────
# # 4. Gera/atualiza lockfile se desejar (opcional)
# #    …ou pule para o post-install
# # ──────────────────────────────────────────────────────────────
# echo "📂 Entrando no projeto: $PROJECT_DIR"
# cd "$PROJECT_DIR" || { echo "❌ Diretório do projeto não encontrado"; exit 1; }

# echo "✅ Setup-script finalizado –  ambiente 'riskpilot-lock' pronto."




# set -e

# ENV_PATH="$HOME/envs/riskpilot-lock"
# PROJECT_DIR="${PROJECT_DIR:-$HOME/RiskPilot}"

# # Detect Python executable
# PYTHON_EXEC=$(command -v python3.11 || command -v python3 || true)
# if [[ -z "$PYTHON_EXEC" ]]; then
#   echo "❌ Python 3 not found. Install Python 3.11 or 3.10 first."
#   exit 1
# fi

# echo "🧹 Removing previous env (if any)..."
# rm -rf "$ENV_PATH"

# echo "🐍 Creating new virtualenv at $ENV_PATH using $PYTHON_EXEC ..."
# "$PYTHON_EXEC" -m venv "$ENV_PATH"

# echo "✅ Activating environment..."
# source "$ENV_PATH/bin/activate"

# echo "🚀 Upgrading build tools (pip, setuptools, wheel)..."
# python -m pip install --upgrade pip setuptools wheel

# echo "📦 Installing pip 23.3.2 + pip‑tools 7.3.0 ..."
# python -m pip install pip==23.3.2 pip-tools==7.3.0

# echo "📂 Entering project directory: $PROJECT_DIR"
# cd "$PROJECT_DIR" || { echo "❌ Project directory not found"; exit 1; }


# não precisa criar requirements.lock em outras maquinas, só na máquina de desenvolvimento

# echo "🔒 Generating requirements.lock (extras: viz, binning)..."
# pip-compile --extra viz --extra binning --generate-hashes --output-file requirements.lock pyproject.toml

# echo "✅ Environment 'riskpilot-lock' ready and requirements.lock generated!"
