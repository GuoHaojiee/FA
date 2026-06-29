#!/bin/bash
set -euo pipefail
ulimit -n 65536

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_MODE=sim
SOC_VERSION=Ascend950PR_9599
CANN_HOME="/home/guohaojie/Ascend/cann-9.0.0"
OUTPUT_DIR="${SCRIPT_DIR}/op_test_template/msprof_output"

echo "[RUN.SH] RUN_MODE=${RUN_MODE}, SOC_VERSION=${SOC_VERSION}"

# Source CANN environment
if [ -f "${CANN_HOME}/set_env.sh" ]; then
    source "${CANN_HOME}/set_env.sh"
fi
export ASCEND_HOME_PATH="${CANN_HOME}"

export LD_LIBRARY_PATH="${CANN_HOME}/tools/simulator/${SOC_VERSION}/lib:${CANN_HOME}/x86_64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[RUN.SH] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# Build
rm -rf build
mkdir build && cd build
cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} ..
make -j$(nproc)
cd "$SCRIPT_DIR"

echo ""
echo "[RUN.SH] Build complete. Binary: op_test_template/test_fa_builtin"
echo "[RUN.SH] Starting msprof op simulator..."
echo ""

mkdir -p "$OUTPUT_DIR"
chmod 750 "$OUTPUT_DIR"

# Run with msprof op simulator — dumps/bin/CSV go under OUTPUT_DIR
LOG_FILE="${OUTPUT_DIR}/run.log"

msprof op simulator \
    --soc-version="${SOC_VERSION}" \
    --output="${OUTPUT_DIR}" \
    --aic-metrics=PipeUtilization \
    --timeout=300 \
    "${SCRIPT_DIR}/op_test_template/test_fa_builtin" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "[RUN.SH] Done. Log: ${LOG_FILE}"
echo "[RUN.SH] Results in: ${OUTPUT_DIR}"
find "$OUTPUT_DIR" -maxdepth 4 \( -name "*.csv" -o -name "*.bin" -o -name "*.json" \) | sort
