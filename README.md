🎗️ Breast Cancer Prediction using Machine Learning

This project applies various machine learning algorithms to predict the likelihood of breast cancer based on diagnostic features. The goal is to build an accurate classification model that can assist in early detection.
📊 Dataset

The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI Machine Learning Repository. It contains 30 numeric features derived from digitized images of fine needle aspirate (FNA) of breast masses.
🧠 Models Trained

Several classification models were implemented and compared:

    Logistic Regression

    K-Nearest Neighbors (KNN)

    Decision Tree Classifier

    Support Vector Machine (SVM)

🏆 SVM achieved the highest accuracy among all the trained models, making it the top-performing classifier for this task.
✅ Workflow

    Data Preprocessing

        Handling missing values (if any)

        Feature normalization using standard scaling

        Train-test split (e.g., 80-20)

    Model Training & Evaluation

        Training the models on the training set

        Evaluating using accuracy, precision, recall, F1-score

        Comparing model performance

    Model Selection

        SVM chosen based on best accuracy and balanced metrics

📈 Results

    SVM outperformed other models in terms of accuracy and robustness.

    Evaluation metrics were visualized using confusion matrices and classification reports.

🚀 How to Run

    Clone the repository

    Install required packages:

    pip install -r requirements.txt

    Open the Jupyter notebook and run all cells

📂 Folder Structure

breast-cancer-prediction/
│
├── breast_cancer_prediction.ipynb   # Main notebook
├── requirements.txt                 # Required Python packages
├── README.md                        # Project summary
└── dataset/                         # Data files (if included)

🔍 Future Enhancements

    Add model interpretability tools (e.g., SHAP or LIME)

    Deploy as a web app using Streamlit

    Experiment with ensemble models like Random Forest or XGBoost

👩‍🔬 Author

Created as part of a healthcare AI initiative to demonstrate predictive modeling in breast cancer screening.
