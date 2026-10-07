# DL-ANN-Churn

A deep learning-powered customer churn prediction project built with TensorFlow and Streamlit. This repository trains an Artificial Neural Network (ANN) to predict whether a bank customer is likely to churn and exposes the model through an interactive web application.

## Overview

Customer churn is a major concern for financial institutions because retaining existing customers is typically more cost-effective than acquiring new ones. This project uses customer demographic and account data to build a predictive model that estimates the probability of churn.

The model is trained on a churn dataset and packaged into a reusable prediction app that accepts user inputs and returns a churn probability along with a simple risk assessment.

## Features

- ANN-based churn prediction model
- Streamlit web interface for easy interaction
- Preprocessing with scaling for numeric features
- Categorical encoding for geography and gender
- Real-time churn probability prediction
- Risk classification: Low Risk vs High Risk

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- scikit-learn
- pandas
- NumPy
- pickle for model artifact loading

## Project Structure

```text
DL-ANN-Churn/
├── app.py                 # Streamlit application
├── churn_model.keras      # Trained Keras model
├── scaler.pkl             # Saved scaler used during feature preprocessing
├── Churn_Modelling.csv    # Customer churn dataset
├── DL_ANN_1.ipynb         # Training notebook
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Optional repository ignore rules
```

## Dataset

The repository uses a bank customer churn dataset containing features such as:

- Credit score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of products
- Credit card usage
- Active membership status
- Estimated salary

This data is used to train the model to classify customers into churn and non-churn categories.

## Model Workflow

1. Load and preprocess the churn dataset.
2. Encode categorical variables.
3. Scale numeric features using a saved scaler.
4. Train an ANN using TensorFlow/Keras.
5. Save the trained model as `churn_model.keras`.
6. Deploy a Streamlit app that loads the model and scaler to make predictions.

## Installation

Clone the repository:

```bash
git clone https://github.com/pushpenderb65/DL-ANN-Churn.git
cd DL-ANN-Churn
```

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate     # Windows
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`) to use the app.

## Example Prediction

The app lets users input customer details and then predicts probability of churn. A prediction above 50% is labeled as high risk.

## Notes

- The app loads the trained model and scaler from local files.
- The feature preprocessing must remain consistent with the training pipeline to ensure correct predictions.
- The notebook (`DL_ANN_1.ipynb`) contains the model-building and training logic.

## License

This project is currently unlicensed. If you plan to distribute or reuse it publicly, consider adding an appropriate open-source license.

## Author

Pushpender Bhardwaj

## Future Improvements

- Add model evaluation metrics such as accuracy, precision, recall, and ROC-AUC
- Improve model explainability with SHAP or feature importance analysis
- Add deployment support for cloud hosting or Docker
- Extend the app with batch prediction and CSV upload support
