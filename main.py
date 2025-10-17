from datamining.data_processing import load_data_from_zip, preprocess_parkinsons_data, split_and_scale
from datamining.model_training import train_and_evaluate_models
from datamining.evaluation import plot_confusion_matrix, plot_model_comparison
from sklearn.metrics import confusion_matrix
import os
import joblib

def main():
    zip_path = "data/raw/parkinsons.zip"
    dataframes = load_data_from_zip(zip_path)

    parkinsons_df = dataframes.get('parkinsons.data')
    if parkinsons_df is None:
        raise FileNotFoundError("Parkinsons data not found inside the zip file.")

    parkinsons_df = preprocess_parkinsons_data(parkinsons_df)
    X_train, X_test, y_train, y_test = split_and_scale(parkinsons_df)

    accuracy_results, confusion_matrices, models = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    os.makedirs("output", exist_ok=True)

    for model_name, model in models.items():
        if model_name == "Enhanced Custom KNN":
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)

        print(f"Confusion Matrix for {model_name}:")
        print(confusion_matrix(y_test, y_pred))
        plot_confusion_matrix(y_test, y_pred, model_name, output_dir='output')
        print("=" * 50)

    plot_model_comparison(accuracy_results, output_dir='output')

    best_model_name = max(accuracy_results, key=accuracy_results.get)
    best_model_accuracy = accuracy_results[best_model_name]
    print(f"The model with the highest accuracy is: {best_model_name} with an accuracy of {best_model_accuracy:.2f}")

    # Save models except Enhanced Custom KNN
    for model_name, model in models.items():
        if model_name != "Enhanced Custom KNN":
            joblib.dump(model, f"models/{model_name.replace(' ', '_')}.joblib")
            print(f"Saved {model_name} model.")

if __name__ == "__main__":
    main()
