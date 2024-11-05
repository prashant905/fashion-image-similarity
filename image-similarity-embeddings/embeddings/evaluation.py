# fashion_similarity/evaluation.py

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from .recommendation import RecommendationSystem

class Evaluation:
    def __init__(self, recommendation_system: RecommendationSystem, data: pd.DataFrame):
        """
        Initialize the Evaluation class with a recommendation system and data.

        :param recommendation_system: Instance of RecommendationSystem to generate recommendations.
        :param data: DataFrame containing metadata (true labels) for images.
        """
        self.recommendation_system = recommendation_system
        self.data = data

    def generate_predictions(self, top_n: int = 1) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Generate true labels and predicted labels for evaluating the model.

        :param top_n: Number of recommendations to consider for predictions.
        :return: Tuple of dictionaries containing true labels and predicted labels for each category.
        """
        true_labels = {col: [] for col in ['masterCategory', 'subCategory', 'articleType', 'baseColour']}
        pred_labels = {col: [] for col in ['masterCategory', 'subCategory', 'articleType', 'baseColour']}
        
        for idx in range(len(self.data)):
            for col in true_labels.keys():
                true_label = self.data.loc[idx, col]
                true_labels[col].append(true_label)
                
                # Get recommendations
                rec_indices, _ = zip(*self.recommendation_system.get_recommendations(idx, top_n=top_n))
                
                # Get the most common class in the recommendations
                rec_classes = self.data.loc[rec_indices, col]
                pred_label = rec_classes.mode().values[0]  # Most frequent class
                pred_labels[col].append(pred_label)
        
        return true_labels, pred_labels

    def evaluate_metrics(self, true_labels: Dict[str, List[str]], pred_labels: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate precision, recall, and F1-score for each category.

        :param true_labels: Dictionary of true labels for each category.
        :param pred_labels: Dictionary of predicted labels for each category.
        :return: Dictionary containing precision, recall, and F1-score for each category.
        """
        metrics = {}
        
        for col in true_labels.keys():
            precision = precision_score(true_labels[col], pred_labels[col], average="weighted")
            recall = recall_score(true_labels[col], pred_labels[col], average="weighted")
            f1 = f1_score(true_labels[col], pred_labels[col], average="weighted")
            
            metrics[col] = {"precision": precision, "recall": recall, "f1_score": f1}
        
        return metrics

    def plot_confusion_matrix(self, true_labels: List[str], pred_labels: List[str], class_names: List[str], title: str):
        """
        Plot confusion matrix for a particular category.

        :param true_labels: List of true labels for the category.
        :param pred_labels: List of predicted labels for the category.
        :param class_names: List of unique class names in the category.
        :param title: Title for the confusion matrix plot.
        """
        conf_matrix = confusion_matrix(true_labels, pred_labels, labels=class_names)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {title}')
        plt.show()

    def generate_report(self, top_n: int = 1):
        """
        Generate classification report and confusion matrices for each category.

        :param top_n: Number of recommendations to consider for predictions.
        """
        true_labels, pred_labels = self.generate_predictions(top_n=top_n)
        
        # Calculate and display metrics
        metrics = self.evaluate_metrics(true_labels, pred_labels)
        for col, scores in metrics.items():
            print(f"Metrics for {col}:")
            print(f"Precision: {scores['precision']:.2f}")
            print(f"Recall: {scores['recall']:.2f}")
            print(f"F1-Score: {scores['f1_score']:.2f}")
            print("\n")
        
        # Display confusion matrices and classification reports
        for col in true_labels.keys():
            class_names = sorted(self.data[col].unique())
            self.plot_confusion_matrix(true_labels[col], pred_labels[col], class_names, title=col)
            print(f"Classification Report for {col}:")
            print(classification_report(true_labels[col], pred_labels[col], target_names=class_names))
