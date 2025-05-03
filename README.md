
# 🧠 Mental Health App – Data Science Assignment

Welcome to the **Mental Health App** – a data-driven initiative designed to explore, analyze, and model mental health questionnaire responses. This project combines data preprocessing, model training, and result visualization to offer meaningful insights for mental health evaluation.

---

## 📁 Project Structure

```
mentalhealth_app/
│
├── data/
│   ├── raw_data.csv              # Original questionnaire responses
│   └── processed_data.csv        # Cleaned & transformed data
│
├── models/
│   ├── saved_models/
│   │   ├── classifier.pkl        # Trained classification model
│   │   ├── scaler.pkl            # Scaler object for feature normalization
│   │   ├── kmeans.pkl            # KMeans clustering model
│   │   ├── clustering_results.png
│   │   ├── confusion_matrix.png
│   │   ├── elbow_curve.png
│   │   ├── feature_importance.png
│   │   └── ...
│   └── model_trainer.py          # Script to train and evaluate models
│
├── tests/
│   └── test_model.py             # Unit tests for ML model functionality
│
├── utils/
│   ├── __init__.py
│   ├── data_processor.py         # Data cleaning and transformation logic
│   ├── load_env.py               # Environment variable loader
│   ├── mongodb_connector.py      # MongoDB integration and connectivity
│   ├── questionnaires.py         # Contains questionnaire definitions
│   └── recommendations.py        # Generates recommendations based on results
│
├── app.py                        # Streamlit application entry point
└── requirements.txt              # Python dependencies
```

---

## ⚙️ Features

- **Streamlit-based Frontend**: An interactive interface for users to take assessments.
- **Model Training**: Classifier and clustering algorithms with performance metrics.
- **Data Visualization**: Graphical representations for clustering, confusion matrices, and feature importance.
- **Recommendation Engine**: Suggests well-being tips based on questionnaire responses.
- **MongoDB Integration**: For scalable data storage and querying (if enabled).

---

## 🚀 How to Run

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Streamlit App**:
   ```bash
   streamlit run app.py
   ```

3. **Access the App**: Visit `http://localhost:8501` in your browser.

---

## 🧪 Model Testing

Use the test suite to verify model integrity:

```bash
python -m unittest tests/test_model.py
```

---

## 🙏 Acknowledgments

Special thanks to the **BrainAI Team** for providing direction, motivation, and resources to help bring this assignment to life.

---

## 📬 Feedback & Suggestions

Open to feedback, feature requests, or collaborations! Drop a message or open an issue in this repo.
