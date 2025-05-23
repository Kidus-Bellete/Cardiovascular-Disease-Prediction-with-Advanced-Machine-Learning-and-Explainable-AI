# Cardiovascular-Disease-Prediction-with-Advanced-Machine-Learning-and-Explainable-AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-green)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Project Overview  
This project provides an end-to-end machine learning pipeline for predicting mortality in heart failure patients using clinical records. The workflow includes:

- **Exploratory Data Analysis (EDA)** to understand feature distributions and relationships  
- **Feature Engineering** to prepare data for modeling  
- **Machine Learning & Deep Learning** model development  
- **Model Evaluation & Comparison**  
- **Explainable AI (XAI)** techniques to interpret model decisions  

**Key Objectives:**  
✔ Predict mortality risk in heart failure patients  
✔ Compare traditional ML vs. deep learning performance  
✔ Explain model decisions using SHAP, LIME, and Permutation Importance  
✔ Identify the most important clinical risk factors  

---

## 2. Dataset Description  
### **Source:**  
[Heart Failure Clinical Records Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)  

### **Features:**  
| Feature | Description | Type |
|---------|------------|------|
| `age` | Patient age | Numerical |
| `anaemia` | Decrease in red blood cells (1=Yes, 0=No) | Binary |
| `creatinine_phosphokinase` | Level of CPK enzyme (mcg/L) | Numerical |
| `diabetes` | Diabetes status (1=Yes, 0=No) | Binary |
| `ejection_fraction` | Percentage of blood pumped per heartbeat | Numerical |
| `high_blood_pressure` | Hypertension status (1=Yes, 0=No) | Binary |
| `platelets` | Platelet count (kiloplatelets/mL) | Numerical |
| `serum_creatinine` | Level of creatinine in blood (mg/dL) | Numerical |
| `serum_sodium` | Level of sodium in blood (mEq/L) | Numerical |
| `sex` | Gender (1=Male, 0=Female) | Binary |
| `smoking` | Smoking status (1=Yes, 0=No) | Binary |
| `time` | Follow-up period (days) | Numerical |
| `DEATH_EVENT` | Death occurrence (1=Yes, 0=No) | Target |

### **Key Observations from EDA:**  
🔹 **Class Imbalance:** ~32% of patients died (target imbalance)  
🔹 **Top Correlated Features with Death:**  
   - ⬇ **`time` (Negative correlation)** → Shorter follow-up linked to higher mortality  
   - ⬆ **`serum_creatinine`** → Higher levels indicate worse kidney function  
   - ⬇ **`ejection_fraction`** → Lower values indicate weaker heart pumping  

---

## 3. Data Preprocessing & Feature Engineering  

### **Steps:**  
1. **Handling Missing Data:** No missing values found  
2. **Scaling:** Applied `StandardScaler` to normalize numerical features  
3. **Class Balancing:** Used **SMOTE** to oversample minority class (death events)  
4. **Train-Validation-Test Split:**  
   - **70% Training**  
   - **15% Validation** (for hyperparameter tuning)  
   - **15% Test** (final evaluation)  

---

## 4. Model Development  

### **A. Traditional Machine Learning Models**  
| Model | Key Parameters | Best Metric (ROC AUC) |
|-------|---------------|----------------------|
| **Logistic Regression** | `max_iter=1000`, L2 regularization | **0.9048** |
| **Random Forest** | `n_estimators=100`, `max_depth=5` | **0.9297** |

### **B. Deep Learning Models**  
| Model | Architecture | Best Metric (ROC AUC) |
|-------|-------------|----------------------|
| **Small NN** | 2 Dense Layers (32→16→1) | **0.8896** |
| **Medium NN** | 3 Dense Layers (64→32→16→1) | **0.9242** |
| **Large NN** | 4 Dense Layers (128→64→32→16→1) | **0.9361** |
| **1D CNN** | Conv1D + MaxPooling + Dense | **0.7532** |

**Training Details:**  
- Optimizer: **Adam (lr=0.001)**  
- Loss: **Binary Cross-Entropy**  
- Callbacks: **Early Stopping, ReduceLROnPlateau**  
- Batch Size: **16**, Epochs: **50**  

---

## 5. Model Evaluation  

### **Performance Comparison**  
| Model | Test AUC | Recommendation | Use Case |
|-------|----------|----------------|----------|
| **Large NN** | **0.9361** | Research applications | Highest performance |
| **Random Forest** | **0.9297** | **Clinical deployment** | Best balance |
| **Medium NN** | **0.9242** | Alternative option | Good performance |
| **Logistic Regression** | **0.9048** | **Transparent decisions** | High interpretability |
| **Small NN** | **0.8896** | Resource-constrained | Efficient baseline |

### **Key Findings:**  
✅ **Best Clinical Model:** **Random Forest** (optimal balance of performance, stability, interpretability)  
✅ **Highest Performance:** **Large Neural Network** (research applications)  
✅ **Most Interpretable:** **Logistic Regression** (transparent clinical decisions)  

---

## 6. Explainable AI (XAI) Analysis  

### **Methods Used:**  
1. **Permutation Importance** → Model-agnostic feature ranking  
2. **SHAP (SHapley Additive Explanations)** → Feature contribution quantification  
3. **LIME (Local Interpretable Model-agnostic Explanations)** → Instance-level explanations  

### **Multi-Method Consensus Results:**  
| Rank | Feature | Overall Score | Clinical Significance |
|------|---------|---------------|----------------------|
| 1️⃣ | **`time`** | **0.2603** | Follow-up duration critical |
| 2️⃣ | **`serum_creatinine`** | **0.1043** | Renal function indicator |
| 3️⃣ | **`ejection_fraction`** | **0.1034** | Cardiac function measure |
| 4️⃣ | **`age`** | **0.0774** | Primary demographic risk |
| 5️⃣ | **`serum_sodium`** | **0.0545** | Electrolyte balance |

### **XAI Validation:**  
- **Method Agreement:** High correlation (0.877-0.969) between all XAI methods  
- **Universal Consensus:** All 6 models agreed on `time` as most important feature  
- **Clinical Alignment:** Results match established cardiovascular risk factors  

---

## 7. How to Run the Code  

### **Prerequisites**  
- **Python 3.8+**  
- **GPU** (optional, for faster neural network training)  
- **8GB RAM** minimum  

### **Installation**  
```bash
# Clone the repository
git clone https://github.com/yourusername/cardiovascular-prediction-xai.git
cd cardiovascular-prediction-xai

# Install required packages
pip install -r requirements.txt
```

**Required Libraries:**
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.6.0
imbalanced-learn>=0.8.0
shap>=0.40.0
lime>=0.2.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
```

### **For Google Colab Users:**
```python
# Install additional packages in Colab
!pip install shap imblearn lime
```

---

### **Step-by-Step Execution**  

#### **Step 1: Data Loading & EDA**  
```python
# Load the dataset
df = pd.read_csv('path.csv')



# run the Jupyter notebook
jupyter notebook notebooks/01_EDA_Analysis.ipynb
```

#### **Step 2: Data Preprocessing**  
```python
# Apply preprocessing pipeline

# Apply SMOTE for class balancing
oversample = SMOTE(random_state=42)
X, y = oversample.fit_resample(X, y)
```

#### **Step 3: Model Training**  
```python
# Train Traditional ML Models


# Train Neural Networks


# Example: Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
```

#### **Step 4: Model Evaluation**  


```

#### **Step 5: Explainable AI Analysis**  
```python
# Run complete XAI pipeline


# Example: SHAP analysis
import shap
explainer = shap.TreeExplainer(rf_model)


```

---

### **Quick Start (All-in-One)**  
```python
# Run the complete pipeline


# Or use the Jupyter notebook
jupyter notebook notebooks/Complete_Pipeline.ipynb
```



---

### **Expected Outputs**  

#### **Performance Results:**
- Model comparison table with AUC scores
- Training history plots for neural networks
- ROC curves and confusion matrices

#### **XAI Results:**
- Feature importance rankings across all methods
- SHAP summary plots and waterfall charts
- LIME explanation visualizations
- Multi-method consensus heatmap

#### **Clinical Insights:**
- Top 5 risk factors for cardiovascular mortality
- Model deployment recommendations
- Clinical implementation guidelines

---


##  Contributing  


---

## 10. License  




⭐ **Star this repository if you find it helpful!** ⭐
