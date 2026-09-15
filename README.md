# Customer Churn Prediction

A machine learning project that explores customer characteristics and compares five classification algorithms to predict **customer churn**—whether a customer leaves a service. The analysis, preprocessing, training, evaluation, and visualizations are contained in [notebook.ipynb](notebook.ipynb).

## Motivation

Understanding churn can help a business identify customers who may leave and plan retention efforts. This project explores how customer demographics, tenure, charges, contracts, and support relate to churn, while demonstrating an end-to-end classification workflow.

The aim is to learn from the data, compare different modeling approaches, and examine which features a Random Forest uses to make predictions. The repository contains an exploratory notebook, rather than a deployed prediction service.

## Project files

| File | Purpose |
| --- | --- |
| `notebook.ipynb` | Data exploration, preprocessing, model training, evaluation, and plots. |
| `customer_churn_data.csv` | Dataset used by the notebook. |
| `README.md` | Project overview, setup instructions, methodology, and results. |

## Dataset

The included [customer_churn_data.csv](customer_churn_data.csv) contains **1,000 customer records and 10 columns**. Its fields describe a subscription service with internet-service options. The original provider, collection method, license, and whether the data is synthetic are not documented in the repository.

| Column | Description | Role in the notebook |
| --- | --- | --- |
| `CustomerID` | Unique customer identifier. | Included as a model input. |
| `Age` | Customer age. | Numeric input. |
| `Gender` | `Female` or `Male`. | Label-encoded input. |
| `Tenure` | Length of the customer relationship; unit is not documented. | Numeric input. |
| `MonthlyCharges` | Monthly service charge; currency is not documented. | Numeric input. |
| `ContractType` | `Month-to-Month`, `One-Year`, or `Two-Year`. | Converted into three indicator columns. |
| `InternetService` | `Fiber Optic`, `DSL`, or the literal string `None`. | Imputed and label-encoded input. |
| `TotalCharges` | Total charges recorded for the customer. | Explored, but excluded from model inputs. |
| `TechSupport` | Whether technical support is available: `Yes` or `No`. | Label-encoded input. |
| `Churn` | Whether the customer churned: `Yes` or `No`. | Prediction target. |

### Data observations

- **883 customers (88.3%)** have `Churn = Yes`; **117 (11.7%)** have `Churn = No`.
- No duplicate rows were found in the notebook.
- The CSV contains **297 literal `None` values** in `InternetService`. The notebook's default `pd.read_csv()` call interprets these as missing values. No other columns are reported as missing.
- All 297 of those records have `Churn = Yes`. This is an observed association, not evidence that the service category causes churn.
- `Tenure` and `TotalCharges` have a strong positive correlation of approximately **0.895**.

## Requirements and setup

The notebook records **Python 3.13.7** as its execution environment. Exact package versions are not recorded, and the repository does not include a dependency lockfile or `requirements.txt`.

| Dependency | Purpose |
| --- | --- |
| pandas | Read the CSV, inspect data, and transform columns. |
| NumPy | Numerical operations and sorting feature importance values. |
| Matplotlib | Create and customize plots. |
| seaborn | Draw the correlation heatmap and feature importance chart. |
| scikit-learn | Imputation, encoding, splitting, models, and evaluation metrics. |
| JupyterLab | Open and execute the notebook. |

From the project directory, create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies and start JupyterLab:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyterlab
jupyter lab
```

Open `notebook.ipynb`, select the Python environment containing these packages, and run the cells from top to bottom. Keep the CSV alongside the notebook: it is loaded using the relative path `customer_churn_data.csv`. Restart the kernel before a complete rerun because preprocessing changes `df` in place.

## Workflow and reasoning

### 1. Load and inspect the data

The notebook uses `head()`, `info()`, `describe()`, missing-value counts, duplicate checks, and category counts to understand the dataset.

**Why:** These checks establish the data structure and reveal issues that affect preprocessing and evaluation, including missing values and the uneven target distribution.

### 2. Explore distributions and correlations

A heatmap displays correlations between numeric fields, excluding `CustomerID`. A pie chart shows the churn distribution.

**Why:** Correlations help identify related or potentially redundant features. The churn distribution provides context for interpreting accuracy: a model can score highly simply by favoring the majority class.

### 3. Handle missing values

`SimpleImputer(strategy="most_frequent")` is fitted to the full dataframe. In practice, it replaces the 297 parsed missing internet-service values with `Fiber Optic`, the most common observed service.

**Why:** Mode imputation is a simple way to retain all rows and provide complete inputs to the classifiers. However, if `None` means “no internet service,” preserving it as a separate category would better retain the original information.

### 4. Encode categorical inputs

`LabelEncoder` converts `Gender`, the remaining two `InternetService` categories, and `TechSupport` to numeric values. `pd.get_dummies()` converts `ContractType` into three Boolean indicator columns. The target remains the string labels `Yes` and `No`.

**Why:** The classifiers need numeric input features. Binary encoding represents two-category fields compactly; one-hot encoding represents contract types without assigning an artificial numeric ranking.

### 5. Select features

The models use `CustomerID`, `Age`, `Gender`, `Tenure`, `MonthlyCharges`, `InternetService`, `TechSupport`, and the three encoded contract columns.

`TotalCharges` is omitted. Its strong correlation with `Tenure` offers a possible redundancy rationale, but the notebook does not explicitly document why it was excluded. It also does not explain the inclusion of `CustomerID`; an identifier should be reassessed before using the model beyond this exercise.

### 6. Split into training and test sets

The notebook uses `train_test_split(test_size=0.2, random_state=42)`, producing **800 training records** and **200 test records**. The split is not stratified by churn.

**Why:** A held-out test set measures performance on records not used for model fitting. The fixed seed makes the split repeatable for the same data and environment.

### 7. Train and compare models

| Model | Configuration in the notebook | Purpose of the comparison |
| --- | --- | --- |
| Random Forest | 100 trees, `random_state=42` | Capture nonlinear relationships and inspect feature importance. |
| Logistic Regression | Default settings, `random_state=42` | Provide a linear classification baseline. |
| Support Vector Machine | Linear kernel, `random_state=42` | Compare another linear decision-boundary approach. |
| K-Nearest Neighbors | 6 neighbors | Compare predictions based on nearby customer records. |
| Gradient Boosting | 100 estimators, `random_state=42` | Compare an ensemble that successively corrects earlier errors. |

No feature scaling is applied. Tree models generally do not need scaling, but scaling matters for distance-based KNN and can help SVM and Logistic Regression. The notebook's explanation that feature ranges are small is not sufficient justification for all five models.

### 8. Evaluate and inspect feature importance

The Random Forest is evaluated with accuracy, a confusion matrix, and per-class precision, recall, and F1-score. The other four models are compared using accuracy on the same test set. A bar chart ranks Random Forest feature importances.

**Why:** The confusion matrix and per-class metrics show which churn outcomes are correctly identified. Feature importance helps inspect model behavior, although it does not establish causal relationships.

## Recorded results

These values come from the notebook's saved outputs; they have not been independently reproduced for this README.

| Model | Test accuracy |
| --- | ---: |
| Random Forest | 100.0% |
| Gradient Boosting | 100.0% |
| Support Vector Machine | 95.5% |
| Logistic Regression | 92.5% |
| K-Nearest Neighbors | 87.0% |

The saved Random Forest confusion matrix is:

| Actual / Predicted | No churn | Churn |
| --- | ---: | ---: |
| No churn | 23 | 0 |
| Churn | 0 | 177 |

Its saved precision, recall, and F1-score are all **1.00** for both classes. Always predicting `Yes` would achieve **88.5%** accuracy on this particular test set, which is higher than the recorded KNN accuracy.

## Limitations and next steps

- **Validate perfect scores.** The 100% results describe one held-out split. Investigate dataset construction, possible leakage, and performance on independent data before drawing broader conclusions.
- **Fit preprocessing on training data only.** Imputation and encoding currently occur before the split. A scikit-learn pipeline fitted after splitting would keep test-set information out of learned preprocessing.
- **Clarify `None` values.** Confirm whether they represent missing information or customers without internet service, then choose the appropriate treatment.
- **Reassess identifier use.** Remove `CustomerID` and compare performance to check whether the models rely on dataset-specific ordering or identifiers.
- **Improve model comparison.** Use stratified splits and cross-validation, report per-class metrics for every model, and account for the strong class imbalance.
- **Scale appropriate models and resolve convergence.** The saved Logistic Regression output contains a convergence warning after reaching its default iteration limit. Apply suitable scaling and reassess the iteration limit before interpreting its score.
- **Record reproducible dependencies.** Pin package versions after validating a clean execution. Fixed random seeds alone do not guarantee identical results across environments.

The current project provides a starting point for understanding churn classification and the importance of checking preprocessing choices alongside model accuracy.
