import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
import pickle
import os

class ImprovedFakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),  # Include bigrams and trigrams
            min_df=2,
            max_df=0.95
        )
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.pipeline = None
        
    def create_enhanced_dataset(self):
        """Create a more comprehensive dataset with better fake/real examples"""
        
        fake_news_data = [
            # Conspiracy theories
            ("BREAKING: Government Secretly Controls Weather Using Hidden Technology", 
             "Leaked documents reveal that government agencies have been manipulating weather patterns for decades using classified technology. Unnamed sources within the meteorological community confirm this shocking revelation that mainstream media refuses to report."),
            
            # Health misinformation
            ("Doctors HATE This One Simple Trick That Cures Everything", 
             "A miracle cure discovered by a local grandmother has been suppressed by big pharma for years. This natural remedy can cure cancer, diabetes, and heart disease, but medical establishments don't want you to know about it because it would destroy their profits."),
            
            # Celebrity gossip/clickbait
            ("You Won't Believe What This Celebrity Did - Photos Inside!", 
             "Shocking photos have emerged showing a famous celebrity in a compromising situation. Sources close to the star reveal explosive details about their private life that will leave you speechless. Click to see the unbelievable images!"),
            
            # Political misinformation
            ("EXPOSED: Secret Meeting Reveals Plot to Control Elections", 
             "Whistleblower leaks classified footage of political elites meeting in secret to discuss how they plan to manipulate upcoming elections. This bombshell revelation shows the depth of corruption in our political system."),
            
            # Economic fear-mongering
            ("URGENT: Economic Collapse Imminent - Prepare Now or Lose Everything", 
             "Financial experts are warning that a complete economic meltdown is just days away. Insider information suggests that banks are preparing for massive failures. You need to act immediately to protect your savings before it's too late."),
            
            # Pseudoscience
            ("Scientists Discover Aliens Have Been Living Among Us for Decades", 
             "Groundbreaking research has allegedly proven that extraterrestrial beings have been secretly integrated into human society. Government cover-ups have hidden this truth from the public, but brave researchers are finally exposing the reality."),
            
            # Anti-vaccine misinformation
            ("Vaccine Contains Dangerous Chemicals That Change Your DNA Forever", 
             "Independent researchers have discovered that vaccines contain harmful substances designed to alter human genetics. Medical authorities are suppressing this information to continue their dangerous agenda against public health."),
            
            # Technology conspiracy
            ("5G Towers Are Actually Mind Control Devices - Here's Proof", 
             "Leaked engineering documents reveal that 5G technology is not for communication but for controlling human thoughts and behavior. Telecommunications companies are working with government agencies to implement mass surveillance and control."),
        ]
        
        real_news_data = [
            # Scientific research
            ("New Study Shows Mediterranean Diet Reduces Heart Disease Risk by 25%", 
             "According to research published in the New England Journal of Medicine, a comprehensive study of 7,500 participants over five years found that following a Mediterranean diet significantly reduces cardiovascular disease risk. The study was conducted by researchers at Harvard Medical School and funded by the National Institutes of Health."),
            
            # Government policy
            ("Department of Education Announces $2 Billion Investment in STEM Programs", 
             "The U.S. Department of Education officially announced a new initiative to invest $2 billion in science, technology, engineering, and mathematics education programs across public schools. The funding will be distributed over three years to improve laboratory facilities and teacher training."),
            
            # Economic news
            ("Federal Reserve Raises Interest Rates by 0.25% to Combat Inflation", 
             "The Federal Reserve announced a quarter-point increase in the federal funds rate following their two-day policy meeting. Fed Chairman Jerome Powell stated that the decision was made to address persistent inflation concerns while maintaining economic stability."),
            
            # Health research
            ("Clinical Trial Shows 40% Reduction in Alzheimer's Progression with New Drug", 
             "Results from a Phase 3 clinical trial involving 1,800 patients demonstrate that an experimental Alzheimer's drug significantly slows cognitive decline. The study, published in The Lancet, was conducted across 15 medical centers and monitored patients for 18 months."),
            
            # Technology news
            ("Tech Company Reports 15% Increase in Renewable Energy Usage", 
             "Microsoft announced in its annual sustainability report that the company increased its renewable energy consumption by 15% compared to the previous year. The report details specific investments in solar and wind energy projects across their data centers."),
            
            # Environmental science
            ("NASA Satellite Data Confirms Arctic Ice Loss Accelerating", 
             "Analysis of satellite imagery from NASA's Ice, Cloud and land Elevation Satellite shows that Arctic sea ice is declining at a rate of 13% per decade. The findings, published in the journal Nature Climate Change, are based on 40 years of continuous monitoring."),
            
            # Education
            ("University Research Team Develops More Efficient Solar Panel Design", 
             "Engineers at Stanford University have created a new solar panel design that increases energy conversion efficiency by 22%. The research, funded by the Department of Energy, was published in the journal Science and has undergone peer review."),
            
            # Public health
            ("CDC Reports 30% Decrease in Smoking Rates Among Teenagers", 
             "The Centers for Disease Control and Prevention released data showing that teenage smoking rates have dropped to historic lows. The National Youth Tobacco Survey, conducted annually since 1999, surveyed 20,000 students across all 50 states."),
        ]
        
        # Create DataFrame
        data = []
        for headline, content in fake_news_data:
            data.append({
                'headline': headline,
                'content': content,
                'label': 'FAKE',
                'combined_text': f"{headline} {content}"
            })
        
        for headline, content in real_news_data:
            data.append({
                'headline': headline,
                'content': content,
                'label': 'REAL',
                'combined_text': f"{headline} {content}"
            })
        
        return pd.DataFrame(data)
    
    def extract_features(self, text):
        """Extract additional features from text"""
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Fake news indicators
        fake_keywords = [
            'shocking', 'unbelievable', 'exposed', 'secret', 'leaked', 'breaking',
            'you won\'t believe', 'doctors hate', 'they don\'t want you to know',
            'mainstream media', 'cover-up', 'conspiracy', 'hidden truth'
        ]
        
        real_keywords = [
            'study', 'research', 'according to', 'published', 'university',
            'official', 'announced', 'data', 'analysis', 'peer-reviewed',
            'scientists', 'researchers', 'evidence', 'findings'
        ]
        
        text_lower = text.lower()
        features['fake_keyword_count'] = sum(1 for keyword in fake_keywords if keyword in text_lower)
        features['real_keyword_count'] = sum(1 for keyword in real_keywords if keyword in text_lower)
        
        return features
    
    def train_model(self, df):
        """Train the improved model"""
        print("Training improved fake news detection model...")
        
        # Prepare text data
        X_text = df['combined_text'].values
        y = (df['label'] == 'FAKE').astype(int)
        
        # Create pipeline with TF-IDF vectorization and logistic regression
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.model)
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        print("\n=== Model Performance ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X_text, y, cv=5, scoring='accuracy')
        print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return metrics
    
    def predict(self, text):
        """Make prediction on new text"""
        if self.pipeline is None:
            raise ValueError("Model must be trained first")
        
        prediction = self.pipeline.predict([text])[0]
        probability = self.pipeline.predict_proba([text])[0]
        
        return {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': max(probability),
            'fake_probability': probability[1],
            'real_probability': probability[0]
        }
    
    def save_model(self, filepath='models/improved_fake_news_model.pkl'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"Model saved to {filepath}")

def main():
    # Initialize detector
    detector = ImprovedFakeNewsDetector()
    
    # Create enhanced dataset
    print("Creating enhanced dataset...")
    df = detector.create_enhanced_dataset()
    print(f"Dataset created with {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Train model
    metrics = detector.train_model(df)
    
    # Save model
    detector.save_model()
    
    # Test with examples
    print("\n=== Testing Model ===")
    
    test_cases = [
        "SHOCKING: Scientists discover aliens living among us - government cover-up exposed!",
        "New study published in Nature shows that regular exercise reduces heart disease risk by 30% according to Harvard researchers."
    ]
    
    for i, text in enumerate(test_cases):
        result = detector.predict(text)
        print(f"\nTest {i+1}: {text[:50]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
    
    print("\nModel training completed successfully!")
    return metrics

if __name__ == "__main__":
    main()
