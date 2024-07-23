import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(probabilities, labels):
    """ Calculate classification metrics based on probabilities and labels and return them as a dictionary """
    # Convert probabilities to predicted labels
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(labels, predictions)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    
    try: 
        roc_auc = roc_auc_score(labels, probabilities, average='weighted', multi_class='ovr')
    except ValueError as e:
        roc_auc = None
        print(e)
        print("ROC AUC score is not calculated for this run, this may be due to not all classes being present.")

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }