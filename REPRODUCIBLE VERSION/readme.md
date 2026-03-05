🌟 CPR-SAT Reproducible Capsule

Contextual Pictogram Retrieval using Hybrid Linguistic Normalization

🌟 Overview

This reproducible capsule implements the experimental evaluation of the CPR-SAT retrieval framework designed for assistive communication systems.

The system retrieves appropriate communication labels from normalized user queries using a hybrid linguistic normalization and similarity-based ranking approach.

The capsule reproduces the experiments described in the study and includes:

🌟 Hybrid linguistic normalization
🌟 TF-IDF + fuzzy similarity ranking
🌟 Context-aware re-ranking
🌟 Cross-validation evaluation on curated datasets

Two evaluation datasets are included:

• CPR_PPD – Pictogram Prediction Dataset
• CPR_CRD – Contextual Retrieval Dataset

All experiments are deterministic to ensure full reproducibility.

📌 Artifact Evaluation – Usage Guide

	• Download the reproducible capsule from Zenodo (DOI: https://doi.org/10.5281/zenodo.18711386
	).

	• Extract the archive to a local working directory.

	• Install dependencies using the provided Conda environment file.

	• Create the environment: conda env create -f environment.yml.

	• Activate the environment: conda activate cprsat-eval.

	• Run the CRD evaluation: python CRD_Eval.py.

	• Run the PPD evaluation: python PPD_Eval.py.

	• The scripts perform cross-validation and compute ranking metrics automatically.

	• The aggregated results are saved to cprsat_5fold_results.csv.

	• The outputs reproduce the experimental results reported in the manuscript.


📂 Repository Structure
CPR-SAT-Reproducible-Capsule
│
├── CPR_CRD.csv
├── CPR_PPD.csv
├── CRD_Eval.py
├── PPD_Eval.py
├── cprsat_5fold_results.csv
├── environment.yml
├── run.bat
└── README.md

📂 File descriptions

File	Description
CPR_CRD.csv	Contextual Retrieval Dataset
CPR_PPD.csv	Pictogram Prediction Dataset
CRD_Eval.py	Evaluation pipeline for CRD experiments
PPD_Eval.py	Evaluation pipeline for PPD experiments
cprsat_5fold_results.csv	Generated results file
environment.yml	Conda environment specification
run.bat	Batch execution script

The evaluation scripts implement the hybrid ranking framework described in the study. 

CRD_Eval

PPD_Eval

⚙️ System Requirements

The capsule was tested using the following configuration:

⚙️ Python 3.10
⚙️ Conda / Anaconda / Miniconda

Dependencies are provided in:

📄 environment.yml 

environment

Installed packages include:

• pandas
• numpy
• scikit-learn
• rapidfuzz

⚙️ Installation

Create the environment using Conda.

conda env create -f environment.yml
conda activate cprsat-eval

This installs all required dependencies for executing the experiments.

📊 Datasets
📊 CPR_CRD.csv

The Contextual Retrieval Dataset (CRD) evaluates ranking performance under contextual conditions.

Key columns include:

Column	Description
normalized	normalized input query
correct_label	ground truth communication label
time_context	contextual time information
location_context	contextual location information

The dataset is loaded through the dataset loader in the CRD evaluation pipeline. 

CRD_Eval

📊 CPR_PPD.csv

The Pictogram Prediction Dataset (PPD) evaluates top-N ranking performance.

Key columns include:

Column	Description
normalized	normalized query
gt_label	ground truth label

Evaluation uses 5-fold cross-validation. 

PPD_Eval

🌟 Methodology
🌟 Hybrid Linguistic Normalization

The system applies a three-stage normalization pipeline.

1️⃣ Rule-Based Correction

Common lexical errors are corrected using deterministic rules.

Example corrections:

watr → water
mil → milk
pls → please
luv → love
2️⃣ Similarity-Based Autocorrection

If rule correction is insufficient, the query is matched against corpus sentences using RapidFuzz token similarity.

The closest matching sentence is selected when the similarity score exceeds a threshold.

3️⃣ AI Fallback (Disabled)

A placeholder module exists for LLM-based normalization, but it is disabled in this capsule to maintain deterministic reproducibility.

🌟 Retrieval Model

Candidate sentences are ranked using a hybrid similarity score:

Score = 0.7 × TF-IDF similarity
      + 0.3 × Fuzzy token similarity

TF-IDF vectorization uses:

TfidfVectorizer(ngram_range=(1,2))
🌟 Context-Aware Re-Ranking

For CRD experiments, contextual signals are optionally incorporated.

The system provides score boosts when:

• time context matches
• location context matches

This improves ranking for context-dependent communication queries.

🔁 Evaluation Protocol
🔁 CRD Evaluation

Uses 3-fold stratified cross-validation.

Metrics:

📊 Mean Absolute Error (MAE)
📊 Root Mean Squared Error (RMSE)
📊 Precision@1
📊 Precision@3
📊 Precision@5

Run using:

python CRD_Eval.py
🔁 PPD Evaluation

Uses 5-fold cross-validation.

Metrics:

📊 Top-1 accuracy
📊 Top-9 accuracy
📊 Top-18 accuracy
📊 Top-25 accuracy
📊 Top-36 accuracy

Run using:

python PPD_Eval.py

Results are saved to:

cprsat_5fold_results.csv
▶️ Running the Capsule
▶️ Manual Execution

Run each experiment individually:

python CRD_Eval.py
python PPD_Eval.py

▶️ Batch Execution

For Windows systems:

run.bat

This executes both evaluation scripts sequentially.

For Linux systems:

run.sh

This executes both evaluation scripts sequentially.

📊 Expected Output

Example console output for CRD evaluation:
=================================
CPR-SAT Evaluation
=================================


Running PPD Evaluation...

=================================
CPR-SAT 5-Fold Cross Validation
=================================

Fold 1 Results:
Top-1: 0.9095
Top-9: 0.9603
Top-18: 0.9625
Top-25: 0.9713
Top-36: 0.9713

Fold 2 Results:
Top-1: 0.8894
Top-9: 0.9580
Top-18: 0.9712
Top-25: 0.9735
Top-36: 0.9779

Fold 3 Results:
Top-1: 0.8805
Top-9: 0.9403
Top-18: 0.9602
Top-25: 0.9624
Top-36: 0.9646

Fold 4 Results:
Top-1: 0.9270
Top-9: 0.9845
Top-18: 0.9889
Top-25: 0.9889
Top-36: 0.9912

Fold 5 Results:
Top-1: 0.9181
Top-9: 0.9779
Top-18: 0.9845
Top-25: 0.9867
Top-36: 0.9867

=================================
Cross-Validation Summary
=================================

Top-1_mean: 0.9049
Top-1_std: 0.0174
Top-9_mean: 0.9642
Top-9_std: 0.0157
Top-18_mean: 0.9735
Top-18_std: 0.0115
Top-25_mean: 0.9766
Top-25_std: 0.0099
Top-36_mean: 0.9783
Top-36_std: 0.0097

Results saved to cprsat_5fold_results.csv

Running CRD Evaluation...

===== BASELINE (No Context) =====
Fold 1: MAE=1.015, RMSE=3.026, P@1=0.812
Fold 2: MAE=1.218, RMSE=18.488, P@1=0.803
Fold 3: MAE=0.785, RMSE=2.400, P@1=0.803

===== CONTEXT-AWARE =====
Fold 1: MAE=0.107, RMSE=1.284, P@1=0.983
Fold 2: MAE=0.203, RMSE=4.800, P@1=0.982
Fold 3: MAE=0.109, RMSE=1.308, P@1=0.982

🔒 Reproducibility Notes

To ensure reproducibility:

🔒 Random seeds are fixed (random_state=42)
🔒 AI normalization fallback is disabled
🔒 Deterministic normalization rules are used
🔒 Fixed cross-validation splits are applied

All experiments can be reproduced by executing the provided scripts.

📄 License

The CPR_PPD and CPR_CRD datasets, together with the accompanying evaluation scripts and environment configuration files included in this reproducible capsule, are released under the Creative Commons Attribution 4.0 International License.

Under this license, users are permitted to:

✔️ Share — copy and redistribute the material in any medium or format

✔️ Adapt — remix, transform, and build upon the material for any purpose, including research and educational use

Provided that appropriate credit is given to the original authors and the source of the dataset.

The complete artifact is publicly archived on Zenodo and can be accessed using the following DOI:

DOI: https://doi.org/10.5281/zenodo.18711386