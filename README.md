# 🧑‍🤝‍🧑 Personality Prediction App

This project is an end-to-end machine learning application that predicts whether a person is an **Introvert** or an **Extrovert** based on their responses to a set of behavioral questions.  
It includes **data cleaning, model training, and a Streamlit-powered web app** for real-time predictions.  

---

## ✨ Features
- **Data Cleaning & Preprocessing**: Robust pipeline for handling missing values and encoding categorical data.  
- **Model Training**: Compared **Random Forest** and **XGBoost** classifiers.  
- **High-Performance Model**: Final **XGBoost model** achieves **96.60% accuracy** on unseen data.  
- **Interactive Web App**: Simple, user-friendly Streamlit interface for instant predictions.  

---

## 🛠️ Tech Stack
- **Data & ML**: pandas, scikit-learn, xgboost  
- **Web Framework**: streamlit  
- **Plotting**: matplotlib, seaborn  
- **Serialization**: pickle  

---

## 📦 Version Control & Workflow
- **.gitignore rules**:  
  - Exclude large files like `.csv`, `.pkl`, `.h5`, and virtual environments.  
- **Training workflow**:  
  - Train the model using `save_model.py` (generates `.pkl` files).  
 
   
    ```bash
    git commit -m "first commit"
    ```
- **Release management**:  
  - Stable versions are tagged.  
  - Example:  
    ```bash
    git tag -a v1.0.0 -m "update estimators"
    git push origin v1.0.0
## 🚀 Run Locally  

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
python -m venv .venv && source .venv/bin/activate   # (Linux/Mac)  
# .venv\Scripts\activate  (Windows)
pip install -r requirements.txt
python save_model.py
streamlit run app.py
