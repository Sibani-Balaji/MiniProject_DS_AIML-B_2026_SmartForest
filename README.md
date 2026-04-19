# MiniProject_DS_AIML-B_2026_SmartForest
# SmartForest: Forest Fire Risk Prediction Using Environmental Data

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Domain](https://img.shields.io/badge/Domain-Environmental%20Analytics-green) ![SDG](https://img.shields.io/badge/SDG-Goal%2015%3A%20Life%20on%20Land-brightgreen)

---

## Abstract

Forest fires are one of the most destructive natural disasters, causing irreversible ecological and economic damage. This project, **SmartForest**, leverages machine learning and environmental data analytics to predict forest fire risk zones. Using the Algerian Forest Fires Dataset from the UCI Machine Learning Repository, we analyze key environmental parameters such as temperature, relative humidity, wind speed, rainfall, and fire weather indices (FFMC, DMC, DC, ISI, BUI, FWI) to build a predictive model that classifies regions as fire or no-fire risk. The project follows a complete data science pipeline — from data cleaning and exploratory data analysis to model development and result interpretation. Multiple classification algorithms including Logistic Regression, Random Forest, and Decision Tree are evaluated. The final model achieves high accuracy and can serve as a decision-support tool for forest management and early warning systems. This project aligns with UN SDG Goal 15: Life on Land, contributing to sustainable forest conservation efforts.

---

## Problem Statement

Forest fires cause severe ecological damage, threatening biodiversity, carbon sinks, and human settlements. Traditional fire monitoring is reactive. This project analyzes environmental parameters to **proactively predict potential fire risk zones**, enabling early intervention and resource planning.

---

## Dataset Source

- **Name:** Algerian Forest Fires Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset++)
- **Records:** 244 instances (2 regions: Bejaia and Sidi Bel-abbes)
- **Features:** Temperature, RH, Wind Speed, Rain, FFMC, DMC, DC, ISI, BUI, FWI, Class

---

## Methodology / Workflow

```
1. Problem Identification
        ↓
2. Dataset Collection (UCI - Algerian Forest Fires)
        ↓
3. Data Cleaning & Preprocessing
   - Handle missing values
   - Remove duplicate rows
   - Encode target variable (fire / not fire)
        ↓
4. Exploratory Data Analysis
   - Feature correlation
   - Distribution plots
   - Region-wise comparison
        ↓
5. Data Visualization
   - Heatmaps, histograms, box plots, pair plots
        ↓
6. Model Development
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
        ↓
7. Evaluation & Result Interpretation
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
```

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python 3.10 | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | ML model development |
| Jupyter Notebook | Interactive analysis |
| GitHub | Version control & documentation |

---

## Results / Findings

- **Best Model:** Random Forest Classifier
- **Accuracy:** ~97%
- Temperature and FWI (Fire Weather Index) were the strongest predictors of fire occurrence
- Region 2 (Sidi Bel-abbes) showed higher fire frequency during summer months
- Humidity inversely correlated with fire risk

> Full results available in `outputs/results/` and `report/mini_project_report.pdf`

---

## Team Members

| Name | Role | GitHub |
|---|---|---|
| Anamika K T | Data Preprocessing & EDA | @Rover-sp24 |
| Sibani B | Model Development & Visualization | @Sibani_Balaji |

---

## Repository Structure

```
MiniProject/
├── README.md
├── requirements.txt
├── docs/
│   ├── abstract.pdf
│   ├── problem_statement.pdf
│   └── presentation.pptx
├── dataset/
│   ├── raw_data/
│   └── processed_data/
├── notebooks/
│   ├── data_understanding.ipynb
│   ├── preprocessing.ipynb
│   └── visualization.ipynb
├── src/
│   ├── preprocessing.py
│   ├── analysis.py
│   └── model.py
├── outputs/
│   ├── graphs/
│   └── results/
└── report/
    └── mini_project_report.pdf
```

---

## SDG Alignment

This project supports **UN Sustainable Development Goal 15 – Life on Land** by promoting technology-driven solutions for forest protection and biodiversity conservation.

---

*SRM Institute of Science and Technology | Mini Project | Data Science | AIML-B | 2026*
