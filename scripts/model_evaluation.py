import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import cross_val_score
import pickle
import os

def load_model_and_data():
    """Load trained model and test data"""
    # Load model
    with open('models/fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load preprocessor components
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load data
    X = np.load('data/X_features.npy')
    y = np.load('data/y_labels.npy')
    
    return model, vectorizer, label_encoder, X, y

def plot_confusion_matrix(y_true, y_pred, labels=['REAL', 'FAKE']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_proba):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curve(y_true, y_proba):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_importance(model, vectorizer, top_n=20):
    """Analyze most important features for each class"""
    feature_names = vectorizer.get_feature_names_out()
    
    # Get feature log probabilities
    feature_log_prob = model.feature_log_prob_
    
    # Class 0 (REAL) - most indicative words
    real_indices = np.argsort(feature_log_prob[0])[-top_n:]
    real_words = [feature_names[i] for i in real_indices]
    real_scores = feature_log_prob[0][real_indices]
    
    # Class 1 (FAKE) - most indicative words
    fake_indices = np.argsort(feature_log_prob[1])[-top_n:]
    fake_words = [feature_names[i] for i in fake_indices]
    fake_scores = feature_log_prob[1][fake_indices]
    
    # Plot feature importance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Real news features
    ax1.barh(range(len(real_words)), real_scores, color='green', alpha=0.7)
    ax1.set_yticks(range(len(real_words)))
    ax1.set_yticklabels(real_words)
    ax1.set_xlabel('Log Probability')
    ax1.set_title('Top Features for REAL News')
    ax1.grid(True, alpha=0.3)
    
    # Fake news features
    ax2.barh(range(len(fake_words)), fake_scores, color='red', alpha=0.7)
    ax2.set_yticks(range(len(fake_words)))
    ax2.set_yticklabels(fake_words)
    ax2.set_xlabel('Log Probability')
    ax2.set_title('Top Features for FAKE News')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'real_features': list(zip(real_words, real_scores)),
        'fake_features': list(zip(fake_words, fake_scores))
    }

def cross_validation_analysis(model, X, y, cv=5):
    """Perform cross-validation analysis"""
    print("Performing cross-validation analysis...")
    
    # Accuracy scores
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Plot CV scores
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, cv+1), cv_scores, alpha=0.7, color='skyblue')
    plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
                label=f'Mean: {cv_scores.mean():.4f}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy Score')
    plt.title('Cross-Validation Accuracy Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/cv_scores.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cv_scores

def generate_evaluation_report():
    """Generate comprehensive evaluation report"""
    print("=== Fake News Detection Model Evaluation Report ===\n")
    
    # Load model and data
    model, vectorizer, label_encoder, X, y = load_model_and_data()
    
    # Split data for evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Basic metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test Set Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}\n")
    
    # Visualizations
    print("Generating visualizations...")
    
    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # ROC Curve
    plot_roc_curve(y_test, y_proba)
    
    # Precision-Recall Curve
    plot_precision_recall_curve(y_test, y_proba)
    
    # Feature Importance
    feature_analysis = analyze_feature_importance(model, vectorizer)
    
    # Cross-validation
    cv_scores = cross_validation_analysis(model, X, y)
    
    # Save comprehensive report
    report = {
        'test_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'cv_scores': cv_scores.tolist(),
        'feature_analysis': feature_analysis
    }
    
    with open('models/evaluation_report.pkl', 'wb') as f:
        pickle.dump(report, f)
    
    print("\nEvaluation completed! All visualizations and reports saved to 'models/' directory.")
    
    return report

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Generate evaluation report
    report = generate_evaluation_report()
