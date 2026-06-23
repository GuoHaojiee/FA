#!/bin/bash

# CANN CAModel Simulator Test Script
# Target SoC: Ascend950PR_9599

set -e

# Configuration
CANN_ROOT="/home/guohaojie/Ascend/cann-9.0.0"
SIMULATOR_PATH="$CANN_ROOT/x86_64-linux/simulator"
SOC_VERSION="Ascend950PR_9599"
TEST_BINARY="./build/fa_camodel_test"

echo "========================================="
echo "CANN CAModel Simulator Test"
echo "========================================="
echo "SOC Version: $SOC_VERSION"
echo "Test Binary: $TEST_BINARY"
echo ""

# Source CANN environment
echo "Setting up CANN environment..."
source "$CANN_ROOT/set_env.sh"

# Configure simulator library path
export LD_LIBRARY_PATH="$SIMULATOR_PATH/$SOC_VERSION/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$SIMULATOR_PATH/dav_3510/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$SIMULATOR_PATH/dav_3510/camodel:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CANN_ROOT/x86_64-linux/devlib/linux/x86_64:$LD_LIBRARY_PATH"

echo "Library paths configured"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""

# Check if binary exists
if [ ! -f "$TEST_BINARY" ]; then
    echo "Error: Test binary not found: $TEST_BINARY"
    echo "Please run 'cd build && cmake .. && make' first"
    exit 1
fi

# Create output directory
OUTPUT_DIR="./output_fa_simulation"
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "Running FlashAttention Simulator Test"
echo "========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Option 1: Try running with msprof op
echo "Attempting to run with msprof op..."
msprof op \
    --output="$OUTPUT_DIR" \
    --application="$TEST_BINARY" \
    --aic-metrics=ArithmeticUtilization,PipeUtilization,Memory \
    --kernel-name="FlashAttentionScore" \
    --launch-count=2 \
    2>&1 | tee "$OUTPUT_DIR/msprof_op.log" || {
    echo "msprof op failed, trying direct execution..."

    # Option 2: Try direct execution
    echo ""
    echo "Running test binary directly..."
    "$TEST_BINARY" 2>&1 | tee "$OUTPUT_DIR/direct_run.log"
}

echo ""
echo "========================================="
echo "Test execution completed"
echo "========================================="
echo "Check output in: $OUTPUT_DIR"
echo ""

# List generated files
if [ -d "$OUTPUT_DIR" ]; then
    echo "Generated files:"
    ls -lh "$OUTPUT_DIR"
fi
