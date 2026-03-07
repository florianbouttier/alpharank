#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-alpharank}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/environment.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda introuvable. Installe Miniconda/Anaconda puis relance." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Error: fichier ${ENV_FILE} introuvable." >&2
  exit 1
fi

cd "${REPO_ROOT}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[AlphaRank] Mise a jour de l'environnement '${ENV_NAME}'..."
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  echo "[AlphaRank] Creation de l'environnement '${ENV_NAME}'..."
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

echo "[AlphaRank] Verification import package + deps..."
conda run -n "${ENV_NAME}" python -c "import alpharank, numpy, pandas, polars, xgboost, optuna, shap, lxml, html5lib; print('OK')"

echo
echo "Environment pret."
echo "Activation: conda activate ${ENV_NAME}"
echo "Execution:  python scripts/run_backtest.py"
