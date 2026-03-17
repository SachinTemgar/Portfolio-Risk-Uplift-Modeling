# Beyond Risk Scores: Uplift-Driven Financial Intervention for Loan Default Prevention

<p align="center">
  <strong>Applied Case Study: Portfolio Risk Management | Financial Services Simulation</strong>
</p>

<p align="center">
  <a href="https://portfolio-risk-uplift-modeling.streamlit.app">Live Dashboard</a> &nbsp;&bull;&nbsp;
  <a href="#results">Results</a> &nbsp;&bull;&nbsp;
  <a href="#methodology">Methodology</a> &nbsp;&bull;&nbsp;
  <a href="#interactive-dashboard">Dashboard</a> &nbsp;&bull;&nbsp;
  <a href="#how-to-run">How to Run</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Containerized-blue?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen" />
</p>

---

## The Problem

Every lending institution has a collections team. When customers show signs of financial stress, this team calls them, offers restructuring, and tries to prevent default. The standard approach ranks customers by a risk score and calls the highest-risk ones first.

**This approach has a blind spot.**

A risk score predicts who will default. It does not predict who will respond to a call. Two customers with identical risk scores can have entirely different outcomes when contacted: one recovers because of the call, the other defaults regardless. A third customer, rarely discussed, actually gets *worse* when contacted.

Risk scoring cannot distinguish between these three outcomes. Uplift modeling can.

---

## The Solution

This project builds an end-to-end causal inference system that identifies exactly which at-risk customers will recover **specifically because of an intervention** and targets only them.

After analyzing 307,511 loan customers, three distinct groups emerged within the 105,366 at-risk population:

| Group | Customers | Share | Effect of Contact |
|-------|-----------|-------|-------------------|
| **Target Group** | 29,131 | 40.5% | Default drops from 20.75% to 6.84% |
| **Will Recover** | 26,865 | 37.4% | No change. Recovery happens without a call |
| **Do Not Contact** | 15,908 | 22.1% | Default increases from 2.92% to 16.19% |

22.1% of collections calls actively push customers toward default. A traditional risk model has no mechanism to identify this. Only uplift modeling can separate customers who benefit from those who are harmed.

---

## Results

| Metric | Uplift Strategy | Current Strategy (Call All) |
|--------|----------------|-----------------------------|
| Customers called | 29,131 | 71,904 |
| Intervention cost | 1,456,550 CU | 3,595,200 CU |
| Net value recovered | 683,509,550 CU | 352,096,403 CU |
| ROI | 46,927% | 9,793% |

**60% fewer calls. 60% lower cost. 2x more value recovered.**

---

## Project Structure

```
Portfolio-Risk-Uplift-Modeling/
│
├── notebooks/
│   ├── 01_eda.ipynb                        Exploratory analysis and feature engineering
│   ├── 02_power_analysis.ipynb             Power analysis and propensity score matching
│   ├── 03_ab_test.ipynb                    A/B test with stochastic intervention simulation
│   ├── 04_uplift_modeling.ipynb            T-Learner, X-Learner, Causal Forest
│   ├── 05_segmentation.ipynb              Segment analysis and collections priority matrix
│   └── 06_decision_report.ipynb           Final recommendation and rollout plan
│
├── dashboard/
│   ├── app.py                              Streamlit interactive dashboard
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── .streamlit/config.toml
│
├── data/
│   ├── raw/                                Original Home Credit dataset
│   └── processed/                          Cleaned and scored outputs per phase
│
├── report/
│   └── collections_strategy_playbook.pdf   Business deliverable for collections managers
│
├── visuals/                                Saved plots from Phase 1
├── requirements.txt                        Project-wide dependencies
└── README.md
```

---

## Methodology

### Phase 1 | Exploratory Data Analysis

Analyzed 307,511 customers across 122 features. Identified and corrected the DAYS_EMPLOYED anomaly (55,374 records with a placeholder value of 365,243). Dropped 41 features with over 50% missing data. Imputed remaining missing values using group-based median strategy. Engineered three financial ratio features: Debt-to-Income, Credit-to-Goods, and Annuity-to-Age. Identified EXT_SOURCE_2 and EXT_SOURCE_3 as the strongest default predictors. Defined the at-risk population as customers with DTI >= 0.20 (105,366 customers, 8.58% default rate).

### Phase 2 | Power Analysis and Experiment Design

Established economic parameters: 50 CU cost per call, 45% Loss Given Default, resulting in a financial break-even of 0.0143%. Set the Minimum Detectable Effect at 2 percentage points based on operational relevance, not the financial floor. Required sample size: 2,738 per group. Available: 52,683 per group (19.2x surplus).

Simulated treatment assignment using domain logic. Matched 35,952 treatment-control pairs via Propensity Score Matching using Logistic Regression and Nearest Neighbor. All four covariates balanced below SMD 0.005. Full common support confirmed across the 0.15 to 0.76 propensity score range.

### Phase 3 | A/B Test Analysis

Simulated intervention effect stochastically. Each defaulted treatment customer received a personal recovery probability based on EXT_SOURCE scores and Debt-to-Income, reflecting the reality that intervention works better on some customers than others.

| Metric | Value |
|--------|-------|
| Observed effect | 2.86 percentage points |
| Z-statistic | -13.86 |
| P-value | < 0.001 |
| 95% Confidence interval | [2.46 pp, 3.26 pp] |
| Post-experiment power | 1.00 |

The lower bound of the confidence interval exceeds the 2.0 pp MDE. The effect is 197x above financial break-even.

### Phase 4 | Uplift Modeling

Built and compared three causal models:

| Model | AUUC | Qini Coefficient |
|-------|------|------------------|
| T-Learner | 0.1960 | 1,937.8 |
| X-Learner | 0.1403 | 1,413.5 |
| Causal Forest (DML) | 0.0949 | 975.5 |
| Random baseline | 0.0289 | 530.2 |

All three models significantly outperform random targeting. Segmented customers into three quadrants using uplift scores and baseline risk. Decile analysis revealed the top 10% of ranked customers show 46.55 pp uplift while the bottom 10% show -26.08 pp uplift (contacting them increases defaults).

### Phase 5 | Segmentation and Collections Strategy

Analyzed the Target Group across income, age, employment, education, and loan type. Every segment is profitable with no exclusions needed.

Priority matrix crosses uplift score against loan value:

| Priority | Description | Customers | Value per Call |
|----------|-------------|-----------|----------------|
| Rank 1 | High Uplift, High Value | 1,330 | 111,340 CU |
| Rank 2-5 | High Uplift, All Values | 7,300 | 31,000 - 67,000 CU |
| Rank 6-11 | Mid Uplift | 10,800 | 8,000 - 21,000 CU |
| Rank 12-16 | Lower Uplift | 9,700 | 2,000 - 8,000 CU |

Optimal targeting depth: 29,106 of 29,131 customers. Virtually the entire Target Group is worth calling. The real cutoff is the boundary between Target Group and non-Target Group.

### Phase 6 | Decision Report

Evaluated four possible outcomes against the evidence:

| Decision | Condition | Result |
|----------|-----------|--------|
| Do Not Deploy | Cost exceeds recovery | Does not apply (197x above break-even) |
| Extend Experiment | Insufficient power | Does not apply (power = 1.00) |
| Targeted Rollout | Only some segments work | Does not apply (all segments profitable) |
| **Full Rollout** | **All conditions met** | **Recommended** |

Staged deployment plan: 5% canary (weeks 1-2), 20% validation (month 1), 100% full scale (months 2-3).

---

## Interactive Dashboard

The Streamlit dashboard provides real-time scenario modeling for collections managers.

**[Open Live Dashboard](https://portfolio-risk-uplift-modeling.streamlit.app)**

**Features:**
- ROI simulator with adjustable cost per outreach, team capacity, and resource limits
- Model selection between T-Learner, X-Learner, and Causal Forest with live chart updates
- Segment filters by loan type and education level
- Qini curve comparing active model against alternatives and random baseline
- Net profit by decile with optimal cutoff visualization
- Customer segmentation scatter map with quadrant summary cards
- Incremental recovery waterfall chart
- Feature importance (uplift score drivers)
- Target Group persona with demographic breakdown and progress bars
- Model reliability panel (SMD balance, statistical power, A/B test results)
- Collections priority matrix heatmap
- Strategy comparison (uplift vs call-all) with improvement percentage
- Downloadable targeting manifest (CSV) for the collections team

---

## Key Findings

**The intervention works.** 2.86 pp default reduction, p < 0.001, power = 1.00.

**Not everyone benefits.** 40.5% respond positively. 37.4% are unaffected. 22.1% get worse when contacted.

**Uplift targeting beats risk targeting.** 2x more value recovered with 60% fewer calls. Every dollar redirected from Do Not Contact and Will Recover customers to the Target Group generates additional returns.

**Every segment is profitable.** All income bands, age groups, employment types, and education levels deliver positive net value per call. No segment needs exclusion.

**The typical responder** is a 42-year-old working professional with secondary education, earning 121,734 CU, carrying a 708,024 CU loan at 28.5% debt-to-income. Financially stressed but capable of recovery when offered restructuring. Their external credit scores are moderate (0.40 and 0.45), noticeably below the Will Recover group. The intervention works because it reaches someone at a tipping point.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Core | Python, pandas, NumPy, Jupyter |
| Statistical Testing | statsmodels (power analysis, z-test, proportion tests) |
| Machine Learning | scikit-learn (Gradient Boosting, Logistic Regression, Nearest Neighbors) |
| Causal Inference | econml (CausalForestDML), custom T-Learner and X-Learner |
| Visualization | matplotlib, Plotly |
| Dashboard | Streamlit |
| Containerization | Docker, Docker Compose |

---

## How to Run

**Prerequisites:** Python 3.11+

**Clone the repo:**
```bash
git clone https://github.com/SachinTemgar/Portfolio-Risk-Uplift-Modeling.git
cd Portfolio-Risk-Uplift-Modeling
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run notebooks:**
```bash
jupyter notebook
```
Open any notebook in the `notebooks/` folder. Run them in order (01 through 06) for the full analysis.

**Run dashboard locally:**
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

**Run dashboard with Docker:**
```bash
cd dashboard
docker compose up --build
```

Dashboard opens at `http://localhost:8501`

---

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) from Kaggle. 307,511 customers across 122 features.

The treatment column is simulated using domain logic, not from a randomized controlled trial. Selection bias was corrected through Propensity Score Matching (all covariates balanced below SMD 0.005). All findings are framed as directional evidence for strategy optimization, consistent with how lending institutions use retrospective causal analysis in practice.

---

## Deliverables

| Deliverable | Location | Audience |
|-------------|----------|----------|
| 6 Jupyter notebooks | `notebooks/` | Data science team |
| Streamlit dashboard | `dashboard/` + [live link](https://portfolio-risk-uplift-modeling.streamlit.app) | Collections manager |
| Collections Strategy Playbook | `report/` | Non-technical stakeholders |
| Dockerized deployment | `dashboard/Dockerfile` | Engineering team |
| Processed datasets | `data/processed/` | Downstream analysis |

---

## Author

**Sachin Temgar**

[GitHub](https://github.com/SachinTemgar) &nbsp;&bull;&nbsp; [LinkedIn](https://www.linkedin.com/in/sachintemgar)

---
