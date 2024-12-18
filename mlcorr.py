import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

def train_svm_classifier():
    # Load data
    file_path = 'workload_datacorr.xlsx'
    df = pd.read_excel(file_path)
    
    # Features and labels
    X = df[['SVM_Output_Workload']].values
    y = df['DL_Output_Workload'].astype(int).values

    # Scaling features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.01, 0.1, 1, 'auto'],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_svm_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

    # Predictions
    y_pred = best_svm_model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Confusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y), annot_kws={"size": 14})
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel("Actual", fontsize=14)
    plt.xlabel("Predicted", fontsize=14)
    plt.show()

    # Plot Precision, Recall, F1-Score, and Accuracy
    metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    scores = [precision, recall, f1, accuracy]

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=metrics, y=scores, palette='viridis')
    plt.title("Precision, Recall, F1-Score, and Accuracy")
    plt.ylim(0, 1)

    # Display values on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 8),
                    textcoords='offset points')

    plt.show()

# Run SVM training and evaluation
train_svm_classifier()
