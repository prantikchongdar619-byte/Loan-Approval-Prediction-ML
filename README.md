# Loan Approval Prediction

A machine learning system to predict loan default risk using customer financial profiles. Built on the Kaggle Loan Approval dataset (58,645 rows, 13 columns).

## Problem Statement

Lenders face two competing risks: approving loans to customers who will default (financial loss), and rejecting customers who would have repaid (lost revenue). This model quantifies default probability per customer and supports three distinct business strategies via adjustable classification thresholds.

## Dataset

**Source**: [Kaggle Loan Approval Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e10)

| Column | Description |
|--------|-------------|
| `person_age` | Applicant age |
| `person_income` | Annual income |
| `person_emp_length` | Years of employment |
| `loan_amnt` | Loan amount requested |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan-to-income ratio |
| `loan_grade` | Credit grade (A–G) |
| `loan_intent` | Purpose of loan |
| `person_home_ownership` | Rent / Own / Mortgage |
| `cb_person_default_on_file` | Historical default on record |
| `cb_person_cred_hist_length` | Credit history length |
| `loan_status` | **Target** — 1: Default, 0: No Default |

**Class imbalance**: Non-defaulters occur ~6× more frequently than defaulters.

## Project Structure

```
loan-approval/
├── Loan Approval.ipynb       # Main notebook
├── train.csv                 # Training data
├── test.csv                  # Test data
├── sample_submission.csv     # Submission format
├── requirements.txt
└── README.md
```

## Feature Engineering

- **Binning**: Age, income, employment length, loan amount, and interest rate binned into deciles using percentile-based `qcut`
- **Interaction features**:
  - `income_to_emp_length` — income stability signal
  - `income_to_age` — earning efficiency relative to career stage
  - `loan_amt_times_int_rate` — total repayment burden proxy
- **Encoding**: One-hot encoding for nominal categoricals (`home_ownership`, `loan_intent`, `default_on_file`); label encoding for ordinal `loan_grade`

## Models Trained

| Model | ROC-AUC | PR-AUC | F1 |
|-------|---------|--------|----|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |
| **XGBoost** ✓ | **0.94** | **0.84** | varies by threshold |

Final model selected on ROC-AUC + recall performance under class imbalance.

## Threshold Strategy

Due to class imbalance, `accuracy` is not used as the primary metric. Three business-aligned thresholds are supported:

| Strategy | Threshold | Recall | Precision | F1 | Use Case |
|----------|-----------|--------|-----------|----|----------|
| Conservative | 0.35 | ~95% | lower | — | Minimize defaults at all cost |
| Balanced | 0.50 | ~70% | ~88% | ~79% | General lending |
| Aggressive | 0.85 | lower | ~95% | — | Rapid market expansion |

## Key Findings — Feature Importance

1. **Loan-to-income ratio** — strongest predictor; higher ratio → higher default risk due to repayment burden
2. **Loan grade** — higher grades (E, F, G) correlate strongly with defaults
3. **Home ownership** — renters default more than homeowners, likely reflecting wealth stability
4. **Loan intent** — medical loans show elevated default rates, suggesting unplanned borrowing without stable income
5. **Interest rate** — higher rates compound the repayment burden and correlate with default

## Expected Loss Model

$$EL = PD \times LGD \times EAD$$

Where:
- **PD** (Probability of Default) = model output score
- **LGD** (Loss Given Default) = assumed recovery rate (tested at 0.5, 0.6, 0.7)
- **EAD** (Exposure at Default) = loan amount

At threshold = 0.5, the approved portfolio achieves an **86% approval rate** with estimated **profit of ₹305 Cr** (assuming LGD = 0.6).

## Setup & Usage

```bash
git clone https://github.com/prantikchongdar619-byte/Loan-Approval-Prediction-ML.git
cd Loan-Approval-Prediction-ML
pip install -r requirements.txt
jupyter notebook "Loan Approval.ipynb"
```

Place `train.csv` and `test.csv` in the project root before running.

## Future Work

- Hyperparameter tuning with cross-validation (currently trained on a single 80/20 split)
- SHAP-based explainability dashboard
- Deployment as a REST API (FastAPI) with a Streamlit frontend
- Integrate dynamic LGD estimation per loan category
