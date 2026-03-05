#!/bin/bash

echo "================================="
echo "CPR-SAT Reproducible Capsule"
echo "Running Evaluation Experiments"
echo "================================="

echo ""
echo "Running PPD Evaluation..."
python PPD_Eval.py

echo ""
echo "Running CRD Evaluation..."
python CRD_Eval.py


echo ""
echo "================================="
echo "All experiments completed"
echo "================================="