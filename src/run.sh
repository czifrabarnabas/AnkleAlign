#!/bin/bash
set -e

echo "=========================================="
echo "AnkleAlign Pipeline"
echo "=========================================="

echo ""
echo "Step 1: Running data processing..."
python 01_data_processing.py

echo ""
echo "Step 2: Running model training..."
python 02_train.py

echo ""
echo "Step 3: Running evaluation..."
python 03_evaluation.py

echo ""
echo "Step 4: Running inference..."
python 04_inference.py

echo ""
echo "=========================================="
echo "Pipeline finished successfully."
echo "=========================================="
