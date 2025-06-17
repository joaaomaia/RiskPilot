#!/usr/bin/env bash
# ----------------------------------------------------------------------
# Script: check_lockfile_safety.sh
# Objetivo: verificar se o arquivo requirements.lock contém dados sensíveis
# ----------------------------------------------------------------------

LOCKFILE="./env_setup/requirements.lock"

echo "🔍 Verificando segurança do arquivo $LOCKFILE..."

# 1. Verifica se o arquivo existe
if [[ ! -f "$LOCKFILE" ]]; then
  echo "❌ Arquivo não encontrado: $LOCKFILE"
  exit 1
fi

# 2. Palavras-chave que indicam riscos
PATTERNS=(
  "token"
  "key"
  "secret"
  "passwd"
  "file://"
  "git@"
  "ssh://"
  "http.*:.*@"
)

# 3. Faz a busca e exibe ocorrências
MATCHES=0
for pattern in "${PATTERNS[@]}"; do
  grep -Ein "$pattern" "$LOCKFILE" && MATCHES=1
done

# 4. Resultado final
if [[ "$MATCHES" -eq 0 ]]; then
  echo "✅ Nenhum indício de conteúdo sensível encontrado em $LOCKFILE"
else
  echo "⚠️ Atenção: possíveis conteúdos sensíveis detectados. Revise antes de publicar."
  exit 2
fi
