#!/bin/bash
# Run from: ~/pediatric_sepsis_prediction_PIC_XAI/
# Usage: bash sepsis_ml/setup_project.sh

set -e

echo "Creating sepsis_ml project structure..."

mkdir -p sepsis_ml/diagnostics
mkdir -p sepsis_ml/models
mkdir -p sepsis_ml/results
mkdir -p sepsis_ml/shap
mkdir -p sepsis_ml/figures
mkdir -p sepsis_ml/logs

echo "Folders created."
echo ""
echo "Project root:  $(pwd)/sepsis_ml"
echo "Data source:   $(pwd)/paediatric-intensive-care-database-1.1.0/output2/"
echo ""
echo "Next: pip install -r sepsis_ml/requirements.txt"
