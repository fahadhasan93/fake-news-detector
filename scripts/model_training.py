import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pickle
import os
from data_preprocessing import NewsDataPreprocessor

class FakeNewsClassifier:
    def __init__(self):
        self.model = MultinomialNB(alpha=1.0)
        self.is_trained = False
        
    def train(self, X_train, y_train):
        """Train the Naive Bayes model"""
        print("Training Naive Bayes classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Model training completed!")
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        print("\n=== Model Evaluation ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
        
        print("\n=== Confusion Matrix ===")
        print(confusion_matrix(y_test, y_pred))
        
        return metrics
    
    def save_model(self, filepath='models/fake_news_model.pkl'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/fake_news_model.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

def generate_more_data():
    """Generate additional synthetic data for better training"""
    fake_patterns = [
        "SHOCKING: {} that will blow your mind!",
        "BREAKING: {} - You won't believe what happened next!",
        "EXPOSED: The truth about {} that they don't want you to know",
        "LEAKED: Secret {} information revealed by insider",
        "UNBELIEVABLE: {} story that mainstream media is hiding"
    ]
    
    real_patterns = [
        "Study shows {} according to researchers at {}",
        "Officials confirm {} in recent announcement",
        "New research indicates {} based on {} analysis",
        "Report reveals {} following comprehensive investigation",
        "{} program launched by {} organization"
    ]
    
    topics = [
        "climate change", "technology advancement", "medical breakthrough",
        "economic policy", "education reform", "space exploration",
        "renewable energy", "artificial intelligence", "public health",
        "urban development"
    ]
    
    institutions = [
        "Harvard University", "Stanford Research Institute", "MIT",
        "Johns Hopkins", "Oxford University", "government agencies"
    ]
    
    # Generate fake news
    fake_data = []
    for i in range(50):
        topic = np.random.choice(topics)
        pattern = np.random.choice(fake_patterns)
        headline = pattern.format(topic)
        content = f"This {topic} story contains sensational claims without proper evidence or sources. " \
                 f"The information appears to be designed to generate clicks rather than inform readers."
        fake_data.append({'headline': headline, 'content': content, 'label': 'FAKE'})
    
    # Generate real news
    real_data = []
    for i in range(50):
        topic = np.random.choice(topics)
        institution = np.random.choice(institutions)
        pattern = np.random.choice(real_patterns)
        headline = pattern.format(topic, institution)
        content = f"According to recent research on {topic}, conducted by {institution}, " \
                 f"new findings have been published in peer-reviewed journals. The study methodology " \
                 f"was rigorous and the results have been independently verified."
        real_data.append({'headline': headline, 'content': content, 'label': 'REAL'})
    
    return pd.DataFrame(fake_data + real_data)

def main():
    # Check if preprocessed data exists
    if not os.path.exists('data/X_features.npy'):
        print("Preprocessed data not found. Running preprocessing first...")
        preprocessor = NewsDataPreprocessor()
        df = preprocessor.create_sample_dataset()
        
        # Add more synthetic data
        additional_data = generate_more_data()
        df = pd.concat([df, additional_data], ignore_index=True)
        
        X, y, _ = preprocessor.preprocess_dataset(df)
        preprocessor.save_preprocessor()
        
        # Save data
        os.makedirs('data', exist_ok=True)
        np.save('data/X_features.npy', X.toarray())
        np.save('data/y_labels.npy', y)
    else:
        print("Loading preprocessed data...")
        X = np.load('data/X_features.npy')
        y = np.load('data/y_labels.npy')
    
    print(f"Dataset shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Initialize and train classifier
    classifier = FakeNewsClassifier()
    classifier.train(X_train, y_train)
    
    # Evaluate model
    metrics = classifier.evaluate(X_test, y_test)
    
    # Save model
    classifier.save_model()
    
    # Save metrics
    with open('models/model_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print("\nModel training and evaluation completed!")
    return metrics

if __name__ == "__main__":
    main()
