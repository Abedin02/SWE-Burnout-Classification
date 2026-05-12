# Code Red: SWE Burnout Classification

A machine learning project that predicts developer burnout levels using behavioral, work pattern, and environmental data. The goal is to enable early intervention by identifying at-risk developers before burnout impacts team performance.

---

## Problem Statement

Developer burnout is a silent productivity killer. In high-pressure software teams, burnout leads to declining code quality, missed deadlines, and destroyed work-life balance — yet most organizations only notice after the damage is done.

This project builds a classification model that detects burnout risk early using measurable developer metrics.

---

## Dataset

- **7,000 developer records**
- **11 feature variables** covering work habits, behavioral signals, and environmental factors
- **Target variable:** `burnout_level` — classified as `High`, `Medium`, or `Low`

| Feature | Description |
|---|---|
| `age` | Developer age |
| `experience_years` | Years of professional experience |
| `daily_work_hours` | Average hours worked per day |
| `sleep_hours` | Average hours of sleep per night |
| `caffeine_intake` | Daily caffeine consumption |
| `bugs_per_day` | Number of bugs encountered daily |
| `commits_per_day` | Daily code commits |
| `meetings_per_day` | Number of meetings per day |
| `screen_time` | Daily screen time in hours |
| `exercise_hours` | Daily exercise in hours |
| `stress_level` | Self-reported stress level *(removed — perfectly correlated with target)* |

---

## Exploratory Data Analysis

Key findings from the EDA:

- **Class imbalance:** Medium burnout accounts for ~50% of records, High ~26%, Low ~23%
- **Sleep is the strongest behavioral indicator:** Developers with High burnout average 6.09 hours of sleep vs 7.00 hours for Low burnout
- **Experience level does not protect against burnout** — sleep and workload matter more than seniority
- **Top workload drivers:** bugs per day and daily work hours showed the strongest feature importance scores

---

## Data Preprocessing

- Missing values in numeric features replaced with the **median**
- Missing values in `burnout_level` replaced with the **mode**
- `stress_level` removed due to perfect correlation with the target variable (caused 100% accuracy — a clear data leakage issue)
- Features scaled using `StandardScaler` for KNN (not required for Random Forest)

---

## Models

### K-Nearest Neighbors (KNN)
- Features: all available features excluding `burnout_level` and `stress_level`
- Hyperparameter tuned: K values from 1 to 29
- **Optimal K = 28**
- **Accuracy: 87.0%**

### Random Forest
- 100 trees as baseline, tuned via `n_estimators`
- **Optimal n_estimators = 60**
- **Accuracy: 77.1%**

| Model | Accuracy | Hyperparameter |
|---|---|---|
| KNN | **87.0%** | K = 28 |
| Random Forest | 77.1% | n_estimators = 60 |

**KNN is the recommended model** for this dataset. Despite Random Forest generally outperforming KNN on complex non-linear datasets, the relatively straightforward feature-target relationships in this dataset favored KNN's distance-based approach.

---

## Feature Importance (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | bugs_per_day | 19.7% |
| 2 | daily_work_hours | 18.8% |
| 3 | screen_time | 12.5% |
| 4 | meetings_per_day | 10.8% |
| 5 | sleep_hours | 10.7% |

Burnout is driven primarily by **workload intensity**, not demographic characteristics like age or experience.

---

## Future Work

- Apply **SMOTE** to address class imbalance (~50% Medium burnout)
- Explore **XGBoost** or **neural networks** for potentially higher accuracy
- Incorporate additional features such as remote work status or team size
- Deploy as a real-time burnout risk monitoring tool

---

## Repository Structure

SWE-Burnout-Classification/
│
├── SWE_Burnout_Prediction.ipynb   # Main notebook
└── README.md                      # Project documentation

---

## Tools & Libraries

- Python, Pandas, NumPy
- Scikit-learn (KNN, Random Forest, StandardScaler)
- Matplotlib, Seaborn
