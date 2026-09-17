# 🐾 Multimodal Classification and Regression for Adoption Speed Prediction (PetFinder)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)
![Metric](https://img.shields.io/badge/Metric-QWK%20(Quadratic%20Weighted%20Kappa)-green.svg)

---

## 📌 Authors
* **Álvaro Lorenzo**
* **Marcos Ortiz**
* **Alberto Águila**

---

## 📸 Project Overview

This project addresses the prediction of pet adoption speed (**`AdoptionSpeed`**) by integrating **heterogeneous, multimodal data sources**:
1. **Tabular Data / Metadata**: Age, Breed, Health Status, Fee, Color, Vaccinated, Dewormed, Sterilized, State, Quantity, Gender, Fur Length, Maturity Size, Type, etc.[cite: 1, 2]
2. **Visual Data**: First pet images extracted from the dataset.
3. **Textual Data**: Text descriptions processed using NLP techniques.

The primary objective is to design, implement, and benchmark **hybrid multimodal architectures**, exploring different fusion strategies (**Early Fusion**, **FiLM Modulation**, **Late Fusion/Ensemble**) and comparing **Classification vs. Ordinal Regression** paradigms.

---

## 📊 Evaluation Metric: Quadratic Weighted Kappa (QWK)

Due to class imbalance and the inherent **ordinal nature** of the target variable, the evaluation metric is **Quadratic Weighted Kappa (QWK)**:
* **0**: Very fast adoption (0–7 days).
* **1**: Fast adoption (8–30 days).
* **2**: Medium adoption (31–90 days).
* **3**: Slow adoption (91–100 days).
* **4**: No adoption (100+ days).

> **Why QWK?** QWK penalizes predictions quadratic to their distance from the true category. Misclassifying class `0` as class `1` receives a small penalty, whereas misclassifying class `0` as class `4` receives a severe penalty.

$$\kappa = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}$$

---

## 🔬 Experimental Setup & Methodological Pipeline

### 1️⃣ Data Preprocessing & Feature Engineering
* **Tabular Branch**:
  * **Numerical Features**: Scaled using `StandardScaler`.
  * **Categorical Features**: Encoded via `LabelEncoder` / Target Encoding.
  * **Text Descriptions**: Feature extraction using **TF-IDF Vectorization** (reduced via Truncated SVD / PCA) and sentiment analysis.
* **Visual Branch**:
  * **Image Preprocessing**: Resize to $224 \times 224$, normalization using ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  * **Data Augmentation**: Random Horizontal Flip, Random Rotation, Color Jittering during training.

### 2️⃣ Model Architectures Explored

#### A. Unimodal Baselines
* **Tabular Baseline (Random Forest / XGBoost)**: Trained exclusively on metadata and text features.
* **Visual Baseline (CNN - EfficientNet-B0)**: Transfer Learning with pre-trained ImageNet weights to predict adoption speed directly from pet photos.

#### B. Multimodal Fusion Strategies
* **Early Fusion**: Concatenation of CNN visual embeddings ($1280\text{-d}$) and tabular feature vectors before passing through a unified Multi-Layer Perceptron (MLP).
* **FiLM (Feature-wise Linear Modulation)**: Advanced deep learning architecture where metadata features conditionally modulate visual feature maps via affine transformations ($\text{Feat}_{\text{out}} = \gamma \cdot \text{Feat}_{\text{in}} + \beta$).
* **Late Fusion (Ensemble)**: Independent training of the Tabular and Visual branches, blending output probabilities via weighted averaging (*Soft Voting*).

#### C. The Paradigm Shift: Classification vs. Ordinal Regression
In standard multi-class classification (Cross-Entropy loss), the model treats all errors equally regardless of class order. To better align with QWK:
1. **Continuous Output Regression**: Replaced Softmax/Cross-Entropy with Mean Squared Error (MSE) / Smooth L1 Loss predicting a continuous value $\hat{y} \in [0, 4]$.
2. **Threshold Optimization (Nelder-Mead / Powell Optimization)**: Post-processing step on the validation set to find optimal decision cutoffs $[t_1, t_2, t_3, t_4]$ mapping continuous predictions to discrete classes $\{0, 1, 2, 3, 4\}$.

---

## 📈 Key Results & Experimental Progression

### Summary Performance Comparison

| Model Architecture | Problem Formulation | QWK Score | Insights & Performance Notes |
| :--- | :---: | :---: | :--- |
| **Random Forest (Tabular Only - no text)** | Classification | `0.330` | Strong baseline; metadata carries high predictive signal. |
| **Random Forest (Tabular + TF-IDF)** | Classification | `0.299` | Text features introduced noise without dimensional reduction. |
| **EfficientNet-B0 (Visual Only)** | Classification | `0.256` | Images alone capture partial aesthetic attributes (cute factor, photo quality). |
| **Early Fusion MLP** | Classification | `0.222` | Overfitting due to high feature dimensionality difference. |
| **FiLM (Modulation)** | Classification | `0.319` | Effective conditioning of visual features by metadata. |
| **Late Fusion (Ensemble)** | Classification | `0.365` | Combining distinct output spaces provided immediate boost. |
| **Early Fusion MLP** | **Regression + Thresholds** | `0.261` | 📈 **+17.6%** improvement over classification. |
| **FiLM (End-to-End)** | **Regression + Thresholds** | `0.334` | 📈 **+4.7%** improvement over classification. |
| 🏆 **Late Fusion (Tabular + Vision)** | **Regression + Thresholds** | **`0.403`** | 🏆 **Best Model (+22% boost over RF baseline)** |

---

## 💡 Key Findings & Conclusions

1. **Tabular Data Dominance**: Structured metadata (Age, Health, Breed, Fee) holds significantly greater predictive power for adoption speed than the visual image alone.
2. **Complementary Role of Vision**: While the vision branch underperforms as a standalone predictor ($QWK = 0.256$), its features complement tabular predictions when combined in a **Late Fusion** scheme, helping resolve edge cases.
3. **The Power of Ordinal Regression + Thresholding**: Shifting from multi-class Cross-Entropy to continuous MSE regression with threshold tuning was the single most impactful breakthrough, boosting QWK scores across all architectures (e.g., Late Fusion jumping from `0.365` to **`0.403`**).
4. **Architecture Superiority**: **Late Fusion (Ensemble)** outperformed complex end-to-end Early Fusion and FiLM setups, preventing cross-modal feature interference during gradient updates.

---

## 📁 Repository Structure

```bash
.
├── PPT_ML_DL.pdf          # Presentation slides detailing theoretical background & results
├── Trabajo_ML_DL.html     # HTML export of the main execution notebook with analyses & plots
└── README.md              # Project documentation and summary

## 🛠️ Installation & Execution

# Clone the repository
git clone [https://github.com/your-username/petfinder-adoption-prediction.git](https://github.com/your-username/petfinder-adoption-prediction.git)
cd petfinder-adoption-prediction

# Install required dependencies
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn scipy