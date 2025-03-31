# Optimization-Project

**UniBg - Optimization Course Project** (Master’s Degree in Computer Engineering)

## Project Objectives
The objective of this project is to implement Support Vector Machine (SVM) models as presented in the paper:
[Breast Cancer Survival and Chemotherapy: A Support Vector Machine Analysis](https://www.researchgate.net/publication/2502581_Breast_Cancer_Survival_and_Chemotherapy_A_Support_Vector_Machine_Analysis).

## Dataset
**Breast Cancer Wisconsin (Diagnostic) Dataset**
- **Instances**: 569  
- **Features**: 30  
- **Classes**: Malignant (M), Benign (B)  
- **Dataset Link**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  

## Work Plan
### 1. Feature Selection using Linear SVM
### 2. Tumor Classification using Non-Linear SVM
- Experimentation with different kernel functions:
  - Polynomial
  - Gaussian (RBF)
  - Sigmoid

### 3. Model Optimization
- Grid Search for hyperparameter tuning
- K-fold Cross Validation for performance evaluation

## Software & Tools
- **Software**: MATLAB  
- **Optimization Tools**: CVX, Mosek  

## Implementation Files
- **Gaussian_Kernel.m**: Grid search for the Gaussian kernel  
- **Polynomial_Kernel.m**: Grid search for the Polynomial kernel  
- **Sigmoidal_Kernel.m**: Grid search for the Sigmoidal kernel  
- **Analysis.m**: Feature selection and model comparison with K-fold Cross Validation (K=10).




