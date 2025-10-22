# 🧠 Parkinson's Disease Classification using Machine Learning
A machine learning pipeline to classify Parkinson’s disease patients based on biomedical voice measurements.

The system preprocesses raw data, trains multiple models including an Enhanced Custom KNN, evaluates their performance, and generates confusion matrices and model comparison plots.


# 📁 Project Structure

parkinsons-disease-prediction-using-ml/

│── __init__.py                   # Package initializer

│── datamining/data_processing.py            # Functions to load, preprocess, split, and scale data

│── datamining/model_training.py             # Model definitions, training, and evaluation

│── datamining/evaluation.py                 # Functions to plot confusion matrices and accuracy comparisons

│── data/raw/parkinsons.zip            # Original Parkinson’s dataset

│── data/processed/                    # Preprocessed CSV files

│── output/                           # Generated confusion matrices and model comparison plots

│── requirements.txt                  # Python dependencies

│── main.py                           # Main script to run the full pipeline

│── README.md                         # Project documentation


# ⚙️ Prerequisites
Python 3.10 or later

pip package manager

Install dependencies, run: pip install -r requirements.txt

If using a virtual environment, activate it before running the command above.


# 🚀 Getting Started
## 📥 Step 1: Prepare the Data
Ensure data/raw/parkinsons.zip exists. The pipeline will automatically preprocess the data and save a cleaned version in data/processed/.

## 🖥️ Step 2: Run the Pipeline
Execute the main script: python main.py

This will:

Load and preprocess the dataset.

Split and scale the features.

Train multiple models:

1. Support Vector Machine (SVM)
2. Scikit-learn KNN
3. Enhanced Custom KNN
4. Naive Bayes
5. Logistic Regression

Generate confusion matrix images for each model.

Generate a model accuracy comparison plot.

Save trained models (except the Enhanced Custom KNN) as .joblib files.

## 📊 Step 3: View Results
Confusion matrices and model comparison plots are saved in the output/ folder.

The console will display confusion matrices and report which model achieved the highest accuracy.


# 🧠 Design Overview
## Data Processing
1. Loads data from a zip archive.
2. Drops irrelevant columns.
3. Splits the dataset into train and test sets.
4. Applies standard scaling to features.

## Model Training
1. Trains multiple classification models.
2. The Enhanced Custom KNN includes feature weighting and adaptive distance-based voting.

## Evaluation
1. Generates confusion matrices for all models.
2. Compares model accuracies using bar plots.
3. Saves plots for offline analysis.


# 🧾 Logging
The pipeline prints model evaluation metrics, confusion matrices, and progress messages to the console.

All plots are saved in the output/ directory for easy visualization.


# 🧪 Testing Notes
1. Test the pipeline by running main.py on the raw dataset.
2. Check the output/ folder to verify confusion matrices and accuracy plots.
3. Modify models, hyperparameters, or feature selection to experiment with performance.


# 🧰 Tools and Technologies
| **Component**            | **Technology Used**                                     |
| ------------------------ | ------------------------------------------------------- |
| Language                 | Python 3.10                                             |
| Machine Learning Models  | SVM, KNN, Naive Bayes, Logistic Regression              |
| Plotting / Visualization | Seaborn, Matplotlib                                     |
| Data Handling            | NumPy, Pandas                                           |
| Development              | Python scripts / Jupyter Notebook                       |


# 💻 Optional: Run the Jupyter Notebook
You can also run the included Jupyter notebook for an interactive experience:

1. Open DataMiningFinalProject.ipynb in Jupyter Notebook or Google Colab.
2. Upload the parkinsons.zip file to the notebook environment.
3. Execute the cells step by step to see data preprocessing, model training, and evaluation plots interactively.

Note: The notebook performs the same pipeline as main.py, but provides visual outputs inline for easier exploration.


# 📘 License
This project is intended for academic and research purposes.

You are free to modify and extend it for experiments in machine learning, biomedical data analysis, or model evaluation techniques.
