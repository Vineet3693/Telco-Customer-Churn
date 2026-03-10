# 🚀 Customer Churn Prediction System

## 📋 Project Overview
End-to-End Machine Learning project for predicting customer churn using the Telco Customer Churn dataset. This project is designed to work seamlessly on both **Kaggle Notebooks** and **VS Code** with identical code structure.

## 📊 Dataset Details
- **Name**: Telco Customer Churn
- **Source**: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows**: 7,043 customers
- **Columns**: 21 features
- **Target**: Churn (Yes/No → 1/0)
- **Task**: Binary Classification

## 📁 Project Structure
```
customer-churn-prediction/
│
├── data/
│   ├── raw/                  # Raw dataset
│   │   └── telco_churn.csv
│   └── processed/            # Cleaned and featured data
│       ├── cleaned_data.csv
│       └── featured_data.csv
│
├── notebooks/                # Jupyter notebooks (cell-by-cell)
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda_visualization.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_preprocessing.ipynb
│   ├── 06_model_training.ipynb
│   ├── 07_model_comparison.ipynb
│   ├── 08_ensemble_learning.ipynb
│   ├── 09_hyperparameter_tuning.ipynb
│   └── 10_final_evaluation.ipynb
│
├── src/                      # Production Python scripts
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/                   # Trained models & results
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── model_results.csv
│
├── app/                      # Streamlit application
│   ├── streamlit_app.py
│   └── utils.py
│
├── reports/figures/          # Generated visualizations
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
├── tests/                    # Pytest test suite
│   ├── test_cleaning.py
│   ├── test_features.py
│   └── test_model.py
│
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore
```

## ⚙️ Tech Stack
- **Language**: Python 3.10+
- **ML Libraries**: scikit-learn, xgboost, lightgbm, catboost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Deployment**: Streamlit
- **Model Persistence**: joblib
- **Imbalance Handling**: imbalanced-learn (SMOTE)
- **Testing**: pytest

## 🚀 Quick Start

### Option 1: Kaggle Notebook
1. Fork this repository
2. Upload dataset to Kaggle
3. Run notebooks sequentially from 01 to 10
4. Download trained models from `/kaggle/working/models/`

### Option 2: VS Code / Local
```bash
# Clone repository
git clone <your-repo-url>
cd customer-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset to data/raw/
# Run notebooks or src scripts
jupyter notebook notebooks/01_data_collection.ipynb
```

## 📓 Notebook Workflow

| Notebook | Purpose | Output |
|----------|---------|--------|
| 01_data_collection | Load & explore data | `data/raw/telco_churn.csv` |
| 02_data_cleaning | Fix types, handle missing | `data/processed/cleaned_data.csv` |
| 03_eda_visualization | Visual analysis | `reports/figures/*.png` |
| 04_feature_engineering | Create new features | `data/processed/featured_data.csv` |
| 05_preprocessing | Scale, encode, split | Preprocessor object |
| 06_model_training | Train 14 baseline models | Multiple model files |
| 07_model_comparison | Compare all models | `models/model_results.csv` |
| 08_ensemble_learning | Voting & stacking | Ensemble models |
| 09_hyperparameter_tuning | Grid/Random search | Tuned models |
| 10_final_evaluation | Final metrics & save | `best_model.pkl` |

## 🤖 Models Trained

### Baseline Models
- Logistic Regression
- Naive Bayes
- KNN
- Decision Tree

### Intermediate Models
- Random Forest
- Gradient Boosting
- AdaBoost
- Bagging Classifier
- SVM

### Advanced Models
- XGBoost
- LightGBM
- CatBoost

### Ensemble Models
- Voting Classifier (Hard + Soft)
- Stacking Classifier

## 📊 Evaluation Metrics
Every model is evaluated on:
- ✅ Accuracy
- ✅ Precision
- ✅ **Recall** (Primary Metric)
- ✅ F1 Score
- ✅ ROC-AUC Score
- ✅ Confusion Matrix
- ✅ Classification Report
- ✅ Cross-Validation Score (cv=5)

## 🔧 Feature Engineering
New features created:
- `tenure_group`: Bin tenure into groups (0-12, 12-24, 24-48, 48-72 months)
- `avg_monthly_to_total_ratio`: MonthlyCharges / TotalCharges
- `is_long_term_customer`: tenure > 36 months (1/0)
- `num_services`: Count of all services subscribed
- `has_no_support`: OnlineSecurity==No AND TechSupport==No (1/0)

## ⚖️ Class Imbalance Handling
- **Technique**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Applied**: Training data ONLY
- **Never applied**: Test data
- **Visualization**: Before/after distribution shown

## 🌐 Streamlit App Features

### Pages
1. **Home**: Project overview and dataset info
2. **EDA**: Interactive visualizations
3. **Predict**: Single customer prediction form
4. **Batch Predict**: Upload CSV for bulk predictions
5. **Model Performance**: Comparison tables and charts

### Features
- Load model from `models/best_model.pkl`
- Load preprocessor from `models/preprocessor.pkl`
- Show prediction probability with gauge chart
- Display SHAP feature importance
- Risk level classification: HIGH / MEDIUM / LOW

## 🧪 Testing
Run tests with:
```bash
pytest tests/ -v
```

Tests cover:
- `test_cleaning.py`: Data cleaning functions
- `test_features.py`: Feature engineering functions
- `test_model.py`: Model prediction output validation

## 💾 Model Saving Strategy

### On Kaggle
```python
import joblib
joblib.dump(model, 'models/best_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')
results_df.to_csv('models/model_results.csv', index=False)
```

### Download & Transfer
1. Download `.pkl` and `.csv` files from Kaggle
2. Place in `models/` folder in VS Code
3. Run Streamlit app locally

## 🎯 Hyperparameter Tuning
- **GridSearchCV**: Top 2 performing models
- **RandomizedSearchCV**: XGBoost and LightGBM
- **Cross-validation**: cv=5
- **Primary scoring metric**: Recall

## 📝 Code Style
- ✅ Every function has docstrings
- ✅ Type hints used throughout
- ✅ Exception handling with try/except
- ✅ Progress printing with emojis
- ✅ `random_state=42` everywhere
- ✅ `n_jobs=-1` for parallel processing

## 🔧 Running on Both Platforms

### Path Handling
The code automatically detects the environment:
```python
if os.path.exists('/kaggle/input/telco-customer-churn/'):
    # Kaggle path
    DATA_PATH = '/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'
else:
    # VS Code path
    DATA_PATH = 'data/raw/telco_churn.csv'
```

### No Code Changes Needed
Same notebook runs on both platforms without modifications!

## 📈 Next Steps
1. ✅ Complete all 10 notebooks
2. ✅ Build src/ production scripts
3. ✅ Create Streamlit app
4. ✅ Write comprehensive tests
5. ✅ Push to GitHub
6. 🚀 Deploy Streamlit app

## 🤝 Contributing
Feel free to fork, improve, and submit pull requests!

## 📄 License
MIT License

## 👨‍💻 Author
Built as a comprehensive ML project demonstrating end-to-end production workflow.

---
**Happy Coding! 🚀**
