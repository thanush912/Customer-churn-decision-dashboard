# Customer Churn Decision Dashboard

A three-page interactive business-intelligence dashboard that turns the IBM Telco Customer Churn dataset into clear, actionable decisions for business owners and managers.

Live Demo: [Customer-churn-decision-dashboard](customer-churn-decision-dashboard.streamlit.app)
---

## Problem Statement

Telecom companies lose significant revenue when customers cancel their subscriptions ("churn"). Identifying *who* is likely to leave, *when*, and *why* enables targeted retention strategies that are far more cost-effective than acquiring new customers.

---

## Business Goal

Turn historical churn data into a decision-support tool that:

1. **Surfaces the most important churn signals** at a glance (KPIs).
2. **Pinpoints which services and customer segments** are most associated with churn.
3. **Scores currently active customers** by churn probability so the retention team can prioritise outreach.
4. **Quantifies the financial impact** of three specific retention actions, with adjustable assumptions.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | Telco Customer Churn (IBM sample data) |
| **Source** | [Kaggle: blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Rows** | 7,043 |
| **Columns** | 21 |
| **Target variable** | `Churn` (Yes / No) |

Key columns: `tenure`, `Contract`, `InternetService`, `PaymentMethod`, `TechSupport`, `MonthlyCharges`, `TotalCharges`, `Churn`.

---

## Tech Stack

| Layer | Library / Tool | Version |
|---|---|---|
| **Front-end** | Streamlit | 1.35.0 |
| **Charts** | Plotly | 5.22.0 |
| **Back-end / Data** | pandas | 2.2.2 |
| **Numerics** | NumPy | 1.26.4 |
| **Model** | scikit-learn | 1.5.0 |
| **Language** | Python | 3.9 (tested) |

> **Python version note:** The project was tested on Python 3.9. The pinned versions are expected to work on Python 3.9 to 3.12 but were not tested on 3.10 or newer. They may fail to install on Python 3.13 or newer.

---

## Model

The churn model is a **Logistic Regression** trained inside a single `sklearn.Pipeline`:

- **Preprocessing:** `StandardScaler` for numeric features; `OneHotEncoder` (`handle_unknown='ignore'`) for categorical features, combined with `ColumnTransformer`.
- **Features:** 18 columns: 4 numeric (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`) and 14 categorical.
- **`gender` is intentionally excluded.** It is a protected demographic attribute and adds little predictive value. Using demographic attributes in a customer-action model raises fairness concerns.
- **No class weighting.** Class weighting would raise recall at some cost in accuracy and precision. The default model was kept for simplicity and interpretability.
- **Train / test split:** 80 / 20, stratified, `random_state=42`.

### Test-set metrics (held-out 20%)

| Metric | Value |
|---|---|
| Accuracy | 80.6% |
| Precision | 65.7% |
| Recall | 55.9% |
| ROC-AUC | 0.842 |

**Recall trade-off:** At the default 0.5 threshold the model misses about 44% of actual churners. This is acceptable for an outreach-prioritisation tool, whose job is to rank customers rather than flag every at-risk individual. A lower threshold (adjustable in the sidebar for the high-risk count) catches more churners at the cost of precision.

---

## Key Insights

> The figures below were computed from the dataset during analysis. The dashboard recalculates every number at runtime from the loaded CSV; nothing is hard-coded. All associations are correlational (see Limitations).

- **Overall churn rate: about 26.5%.** Roughly 1 in 4 customers has left.
- **Month-to-month customers churn at about 42.7%**, versus about 2.8% for two-year contracts. Contract type is the strongest association among the drivers examined.
- **Customers in their first 12 months churn at about 47.4%**, the highest-risk tenure group.
- **Fiber optic customers churn at about 41.9%**, and this segment accounts for the largest share of monthly revenue lost to churn.
- **Electronic check payers churn at about 45.3%**, versus about 15% to 17% for automatic payment methods.
- **Customers without tech support churn at about 41.6%**, versus about 15.2% for those with it.

---

## Dashboard Pages

| Page | Contents |
|---|---|
| **1. Executive Overview** | KPI cards, churn by tenure group, contract mix donut, data-driven headline insight |
| **2. Services & Revenue** | Churn by internet service, payment method and tech support; monthly charges distribution; top-5 high-churn combinations |
| **3. Risk, Opportunities & Actions** | Contract x tenure heatmap, revenue lost by segment, estimated revenue at risk, top-20 at-risk customers with CSV download, 3 risk cards, 3 opportunity cards, 3 action cards |

A **sidebar panel** on Page 3 lets you adjust:

- The assumed conversion percentage (5% to 50%) for each of the three retention actions. The formula and assumption are shown under each estimate.
- A **high-risk threshold** (0.30 to 0.80) that controls only the "High-risk active customers" count, independently of the expected-revenue calculation.

---

## How to Install and Run

### 1. Clone this repository

```bash
git clone https://github.com/thanush912/customer-churn-decision-dashboard.git
cd customer-churn-decision-dashboard
```

### 2. Place the dataset

The app loads the **first `.csv` file found** in the `data/` folder, so the file name does not need to match exactly.

```
data/
└── Telco-Customer-Churn.csv    <- any *.csv name works
```

If the folder is empty or missing, the app shows a file-upload widget instead. The dataset can be downloaded from the Kaggle link above.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run churn_dashboard_app.py
```

The dashboard opens automatically at `http://localhost:8501`. If it does not, paste that address into your browser.

---

## Project Structure

```
customer-churn-decision-dashboard/
├── churn_dashboard_app.py   <- single app file (data loading, cleaning, model, all pages)
├── requirements.txt
├── README.md
└── data/
    └── Telco-Customer-Churn.csv
```

---

## Data Cleaning Steps

| Issue | Resolution |
|---|---|
| `TotalCharges` stored as text, with 11 blank values | Converted to a number; blanks set to 0.0 (these are new customers with tenure = 0 and no charges yet) |
| `SeniorCitizen` stored as 0/1 | Added a `SeniorCitizenLabel` column (Yes/No) for display |
| No tenure grouping | Added `TenureGroup`: 0-12, 13-24, 25-48 and 49+ months |
| Duplicate customers | Checked: no duplicate `customerID` values |

---

## Limitations

1. **Correlation, not causation.** Every association shown (for example, fiber optic and high churn) reflects patterns in historical data. External factors such as price changes, competition and service outages are not captured. The dashboard carries a visible note to this effect.
2. **Impact estimates are illustrative.** The revenue figures on Page 3 use the formula `converted_count x avg_segment_monthly_charge x churn_rate_reduction`. Actual results depend on campaign design, customer behaviour and market conditions.
3. **The model is a ranking tool, not a guarantee.** At the 0.5 threshold it misses about 44% of actual churners (recall about 56%). Use the probability scores to prioritise outreach lists, not as final yes/no decisions. A ROC-AUC of 0.842 means the model ranks a randomly chosen churner above a randomly chosen non-churner about 84% of the time.
4. **`gender` is excluded by design.** Even where including it might marginally improve accuracy, using a protected attribute to target customers for retention actions raises fairness concerns.
5. **Static dataset.** The CSV is a snapshot. The dashboard does not connect to a live data source, and churn patterns may change over time.
6. **No full fairness audit.** Excluding `gender` is a first step only. A production deployment should audit model outputs across protected groups before scores are used for customer decisions.

---

*Built as an IBM internship project using IBM Bob, Streamlit, Plotly, pandas and scikit-learn.*
