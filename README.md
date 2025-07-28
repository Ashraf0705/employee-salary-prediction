# 💼 Employee Salary Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://employee-salary-prediction-smash.streamlit.app/)

A machine learning project that classifies an individual's salary as either `<=50K` or `>50K` based on census data. This repository contains the complete end-to-end workflow, from data analysis in a Jupyter Notebook to a deployed, interactive web application built with Streamlit.

---

### 🚀 Live Demo

The final application is deployed and publicly accessible. You can test it here:

**[Live App: Employee Salary Prediction](https://employee-salary-prediction-smash.streamlit.app/)**

### 📋 Project Summary

This project tackles a classic binary classification problem using the "Adult" dataset from the UCI Machine Learning Repository. The goal is to build a predictive model that can accurately determine a person's income bracket.

The project pipeline includes:
1.  **Data Cleaning & Preprocessing:** Handling missing values, removing outliers, and encoding categorical features.
2.  **Model Evaluation:** Training and comparing five different classification models (Logistic Regression, Random Forest, KNN, SVM, and Gradient Boosting).
3.  **Model Selection:** The **Gradient Boosting Classifier** was selected as the best-performing model with an **accuracy of 85.71%**.
4.  **Deployment:** The trained model was saved and integrated into a Streamlit web application, which was then deployed to the cloud.

### ✨ Application Features

-   **Interactive UI:** A user-friendly interface with sliders and select boxes for inputting 13 different employee attributes.
-   **Real-time Predictions:** Instantly classifies the input data and displays the predicted salary class.
-   **Prediction Confidence:** Shows the model's prediction probabilities, indicating how confident it is in its classification.
-   **Responsive Design:** The app is accessible and functional on both desktop and mobile devices.

### 🛠️ Technologies & Libraries

-   **Language:** Python
-   **Data Analysis & ML:** Pandas, Scikit-learn, Matplotlib, Joblib
-   **Web Application:** Streamlit
-   **Version Control:** Git & GitHub

### 📁 Repository Structure
├── 📄 app.py # The main Streamlit application script
├── 📄 best_model.pkl # The saved, trained Gradient Boosting model
├── 📄 requirements.txt # Required Python libraries for deployment
├── 📄 adult.csv # The dataset used for training the model
└── 📄 employee salary prediction.ipynb # Jupyter Notebook with the full data analysis and model training process


### ⚙️ How to Run This Project Locally

To run the application on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/employee-salary-prediction.git
    cd employee-salary-prediction
    ```
    *(Replace `your-username` with your actual GitHub username)*

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

### 📊 Model Performance

The following models were evaluated, with Gradient Boosting showing the best overall performance.

| Model | Accuracy |
| :--- | :---: |
| **Gradient Boosting** | **0.8571** |
| Random Forest | 0.8490 |
| K-Nearest Neighbors (KNN) | 0.8245 |
| Logistic Regression | 0.8149 |
| Support Vector Machine (SVM) | 0.8396 |
