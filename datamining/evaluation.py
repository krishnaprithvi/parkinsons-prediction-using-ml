import matplotlib
# matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir='output'):
    """
    Plots and saves a confusion matrix with proper annotations and automatic contrast.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    
    ax = sns.heatmap(
        cm, 
        annot=False, 
        fmt="d", 
        cmap="Blues", 
        cbar=False,
        xticklabels=['Negative', 'Positive'], 
        yticklabels=['Negative', 'Positive'], 
        linewidths=0.5, 
        linecolor='gray'
    )
    
    # Automatically adjust annotation color based on cell value
    max_val = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            color = 'white' if value > max_val / 2 else 'black'
            ax.text(j + 0.5, i + 0.5, str(value),
                    ha='center', va='center',
                    color=color, fontsize=12)
    
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"cm_{model_name.replace(' ', '_').lower()}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(accuracy_results, output_dir='output'):
    """
    Plots and saves a bar plot comparing model accuracies.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(accuracy_results.keys()), y=list(accuracy_results.values()))
    plt.title('Model Comparison: Accuracies')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.axhline(y=max(accuracy_results.values()), color='r', linestyle='--', label='Highest Accuracy')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    file_path = os.path.join(output_dir, "model_accuracy_comparison.png")
    plt.savefig(file_path)
    plt.close()
