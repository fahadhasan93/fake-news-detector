# 🛡️ Fake News Detection System

An advanced AI-powered web application that detects fake news using Natural Language Processing and Machine Learning techniques. Built with Next.js, TypeScript, and Python.

![Fake News Detector](https://img.shields.io/badge/AI-Powered-purple)
![Next.js](https://img.shields.io/badge/Next.js-15-black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)


## 🎨 Screenshots

<img width="899" height="717" alt="Screenshot From 2025-07-20 10-49-14" src="https://github.com/user-attachments/assets/8c434ca5-72bf-46bd-8d84-5ccea7f385cd" />


## 🌟 Features

- **Real-time Analysis**: Instant fake news detection with confidence scores
- **Advanced NLP**: Multi-factor text analysis including sentiment and structural patterns
- **Interactive UI**: Modern, responsive interface with purple-emerald theme
- **Machine Learning**: Naive Bayes and Logistic Regression models
- **Detailed Analytics**: Comprehensive breakdown of detection factors
- **Test Examples**: Built-in examples to demonstrate functionality
- **Model Training**: Integrated training pipeline with performance metrics

## 🚀 Demo

Try these examples:

**Fake News Example:**
\`\`\`
Headline: "SHOCKING: Scientists Discover Aliens Living Among Us - Government Cover-Up Exposed!"
Result: ❌ FAKE (High Confidence)
\`\`\`

**Real News Example:**
\`\`\`
Headline: "New Study Shows Regular Exercise Reduces Risk of Heart Disease by 30%"
Result: ✅ REAL (High Confidence)
\`\`\`

## 🛠️ Tech Stack

### Frontend
- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - Modern component library
- **Lucide React** - Beautiful icons

### Backend
- **Next.js API Routes** - Server-side endpoints
- **Python** - Machine learning processing
- **scikit-learn** - ML algorithms and tools
- **NLTK** - Natural language processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Machine Learning
- **TF-IDF Vectorization** - Text feature extraction
- **Logistic Regression** - Primary classification model
- **Naive Bayes** - Alternative classification approach
- **Cross-validation** - Model validation
- **Feature Engineering** - Advanced text analysis

## 📦 Installation

### Prerequisites
- Node.js 18+ 
- Python 3.8+
- npm or yarn

### 1. Clone the Repository
\`\`\`bash
git clone https://github.com/fahadhasan93/fake-news-detector.git
cd fake-news-detector
\`\`\`

### 2. Install Dependencies

**Frontend Dependencies:**
\`\`\`bash
npm install
# or
yarn install
\`\`\`

**Python Dependencies:**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Set Up Python Environment (Recommended)
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
\`\`\`

### 4. Run the Application
\`\`\`bash
npm run dev
# or
yarn dev
\`\`\`

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📋 Requirements File

Create a \`requirements.txt\` file:
\`\`\`
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk>=3.8
matplotlib>=3.6.0
seaborn>=0.12.0
\`\`\`

## 🧠 How It Works

### 1. Text Preprocessing
- HTML tag removal
- Text normalization (lowercase, punctuation)
- Stop word filtering
- Stemming with Porter Stemmer

### 2. Feature Extraction
- **TF-IDF Vectorization**: Converts text to numerical features
- **Linguistic Analysis**: Sentiment, structure, and pattern detection
- **Multi-factor Scoring**: Weighted indicators for fake vs real news

### 3. Classification
- **Primary Model**: Logistic Regression with TF-IDF features
- **Backup Model**: Multinomial Naive Bayes
- **Confidence Scoring**: Probability-based confidence calculation

### 4. Detection Indicators

**Fake News Indicators:**
- Sensational language ("shocking", "unbelievable")
- Clickbait phrases ("you won't believe", "doctors hate")
- Conspiracy terms ("cover-up", "exposed", "leaked")
- Emotional manipulation
- Vague sources ("unnamed sources", "insider reveals")

**Real News Indicators:**
- Credible sources ("according to", "study shows")
- Official language ("announced", "confirmed")
- Specific details (statistics, data, research)
- Professional tone
- Factual reporting structure

## 📊 Model Performance

- **Accuracy**: ~92-97%
- **Precision**: ~89-97%
- **Recall**: ~91-97%
- **F1-Score**: ~90-97%

*Performance metrics based on training dataset and cross-validation*

## 🎯 Usage

### Web Interface
1. **Enter Text**: Input news headline and/or content
2. **Analyze**: Click "Analyze News Article"
3. **Review Results**: See classification, confidence, and detailed analysis
4. **Test Examples**: Use built-in examples to test functionality

### API Endpoints

**Predict News Authenticity:**
\`\`\`bash
POST /api/predict
Content-Type: application/json

{
  "headline": "News headline here",
  "content": "News content here"
}
\`\`\`

**Train Model:**
\`\`\`bash
POST /api/train
\`\`\`

**Process Data:**
\`\`\`bash
POST /api/process-data
\`\`\`

## 🔧 Configuration

### Environment Variables
Create a \`.env.local\` file:
\`\`\`
# Add any required environment variables here
NODE_ENV=development
\`\`\`

### Model Configuration
Modify model parameters in \`scripts/improved_model_training.py\`:
\`\`\`python
# TF-IDF Configuration
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95
)

# Model Configuration
model = LogisticRegression(
    random_state=42,
    max_iter=1000
)
\`\`\`




## 🧪 Testing

### Run Python Scripts
\`\`\`bash
# Process data
python scripts/data_preprocessing.py

# Train model
python scripts/improved_model_training.py

# Evaluate model
python scripts/model_evaluation.py
\`\`\`

### Test API Endpoints
\`\`\`bash
# Test prediction
curl -X POST http://localhost:3000/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{"headline": "Test headline", "content": "Test content"}'
\`\`\`

### Development Guidelines
- Follow TypeScript best practices
- Use ESLint and Prettier for code formatting
- Write meaningful commit messages
- Add tests for new features
- Update documentation

## 📈 Future Enhancements

- [ ] **BERT Integration**: Implement transformer-based models
- [ ] **Real Dataset**: Connect to larger, real-world datasets
- [ ] **User Authentication**: Add user accounts and history
- [ ] **API Rate Limiting**: Implement request throttling
- [ ] **Batch Processing**: Support multiple article analysis
- [ ] **Export Features**: PDF/CSV report generation
- [ ] **Mobile App**: React Native version
- [ ] **Browser Extension**: Chrome/Firefox extension

## 🐛 Known Issues

- Model performance depends on training data quality
- Limited to English language content
- Requires internet connection for full functionality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

[My GitHub](https://github.com/fahadhasan93)


## 🙏 Acknowledgments

- **scikit-learn** - Machine learning library
- **NLTK** - Natural language processing toolkit
- **Next.js** - React framework
- **Vercel** - Deployment platform
- **shadcn/ui** - Component library


---

⭐ **Star this repository if you found it helpful!**
\`\`\`

Made with ❤️ and AI
\`\`\`
