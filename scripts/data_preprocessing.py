import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class NewsDataPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.label_encoder = LabelEncoder()
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stop words
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Apply stemming
        words = [self.stemmer.stem(word) for word in words]
        
        return ' '.join(words)
    
    def create_sample_dataset(self):
        """Create a sample dataset for demonstration"""
        fake_news = [
            ("BREAKING: Scientists Discover Aliens Living Among Us!", 
             "In a shocking revelation that will change everything we know about humanity, scientists have allegedly discovered that aliens have been living among us for decades. This unbelievable story comes from unnamed sources who claim to have inside information about a government cover-up."),
            ("You Won't Believe What This Celebrity Did Next!", 
             "This shocking story about a famous celebrity will leave you speechless. Sources close to the star reveal secrets that the mainstream media doesn't want you to know. The truth has finally been exposed!"),
            ("Secret Government Program Exposed by Whistleblower", 
             "A mysterious whistleblower has leaked classified documents revealing a secret government program that has been operating in the shadows. This explosive revelation shows how deep the conspiracy goes."),
            ("Miracle Cure Discovered - Doctors Hate This One Trick", 
             "Local woman discovers amazing cure that doctors don't want you to know about. This simple trick has been suppressed by big pharma for years, but now the secret is out."),
            ("Breaking: Economy Will Collapse Tomorrow According to Expert", 
             "Financial expert predicts total economic collapse within 24 hours. This shocking prediction is based on secret insider information that the government is trying to hide from the public.")
        ]
        
        real_news = [
            ("New Study Shows Benefits of Regular Exercise", 
             "According to a comprehensive study published in the Journal of Health Sciences, regular exercise has been shown to improve cardiovascular health and mental well-being. The research, conducted over five years with 10,000 participants, provides strong evidence for the benefits of physical activity."),
            ("Local School District Announces New STEM Program", 
             "The Springfield School District officially announced the launch of a new STEM education program starting next fall. The program, funded by a federal grant, will provide students with hands-on experience in science, technology, engineering, and mathematics."),
            ("Research Team Develops New Solar Panel Technology", 
             "Scientists at the University of Technology have developed a new type of solar panel that is 25% more efficient than current models. The research, published in Nature Energy, could significantly impact renewable energy adoption."),
            ("City Council Approves Infrastructure Improvement Plan", 
             "The city council voted unanimously to approve a $50 million infrastructure improvement plan. The plan includes road repairs, bridge maintenance, and upgrades to the water treatment facility, with work scheduled to begin next spring."),
            ("Medical Study Confirms Effectiveness of New Treatment", 
             "A clinical trial involving 500 patients has confirmed the effectiveness of a new treatment for diabetes. The study, conducted at multiple medical centers, showed significant improvement in patient outcomes with minimal side effects.")
        ]
        
        # Combine datasets
        data = []
        for headline, content in fake_news:
            data.append({
                'headline': headline,
                'content': content,
                'label': 'FAKE'
            })
        
        for headline, content in real_news:
            data.append({
                'headline': headline,
                'content': content,
                'label': 'REAL'
            })
        
        return pd.DataFrame(data)
    
    def preprocess_dataset(self, df):
        """Preprocess the entire dataset"""
        print("Starting data preprocessing...")
        
        # Combine headline and content
        df['message'] = df['headline'].fillna('') + ' ' + df['content'].fillna('')
        
        # Clean the combined text
        print("Cleaning text data...")
        df['cleaned_message'] = df['message'].apply(self.clean_text)
        
        # Remove empty messages
        df = df[df['cleaned_message'].str.len() > 0]
        
        # Encode labels
        print("Encoding labels...")
        df['label_encoded'] = self.label_encoder.fit_transform(df['label'])
        
        # Vectorize text
        print("Vectorizing text...")
        X = self.vectorizer.fit_transform(df['cleaned_message'])
        y = df['label_encoded'].values
        
        print(f"Dataset shape: {X.shape}")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return X, y, df
    
    def save_preprocessor(self, filepath='models/'):
        """Save the preprocessor components"""
        os.makedirs(filepath, exist_ok=True)
        
        with open(f'{filepath}/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(f'{filepath}/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Preprocessor saved to {filepath}")

def main():
    # Initialize preprocessor
    preprocessor = NewsDataPreprocessor()
    
    # Create sample dataset
    print("Creating sample dataset...")
    df = preprocessor.create_sample_dataset()
    
    # Preprocess data
    X, y, processed_df = preprocessor.preprocess_dataset(df)
    
    # Save preprocessed data
    os.makedirs('data', exist_ok=True)
    processed_df.to_csv('data/processed_news_data.csv', index=False)
    
    # Save feature matrix and labels
    np.save('data/X_features.npy', X.toarray())
    np.save('data/y_labels.npy', y)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("Data preprocessing completed successfully!")
    print(f"Processed {len(processed_df)} samples")
    print(f"Feature matrix shape: {X.shape}")

if __name__ == "__main__":
    main()
