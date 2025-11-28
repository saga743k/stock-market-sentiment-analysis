# 📈 Stock News Sentiment Analyzer

A desktop application that predicts the sentiment (Positive/Negative) of stock market news headlines using a **Deep Learning LSTM model with GloVe embeddings**.  
The model is integrated into a full-screen **Tkinter GUI** and supports **multiple headlines** at once.

---

## 🚀 Features
- Full-screen Tkinter desktop app  
- Enter one or multiple headlines (one per line)  
- Real-time sentiment prediction  
- Pre-trained GloVe embeddings  
- LSTM deep learning architecture  
- Confidence score + raw probability  
- Scrollable results window  
- Save predictions to `.txt` file  
- Proper error handling and logs  

---

## 🧠 Machine Learning Model

This project uses a deep learning architecture trained on stock news headlines.

### Model Components:
- **Tokenizer + Padding (length = 30)**
- **GloVe word embeddings** (100d/200d)
- **LSTM layer** to capture sequence meaning  
- **Dense + Dropout layers** for refinement  
- **Sigmoid output layer** (Logistic Regression)  

### Model Flow:
```
Input → GloVe Embedding → LSTM → Dense → Sigmoid → Sentiment Output
```

---

## 📚 Technologies Used

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- NLTK  
- Tkinter (GUI)  
- Regex (text cleaning)  
- Virtual Environment (venv)  

---

## 📂 Project Structure
```
SentimentApp/
│── app.py
│── sentiment_model.keras
│── tokenizer.pkl
│── requirements.txt
│── README.md
```

---

## 🛠 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a virtual environment

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK stopwords
```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 5. Run the app
```bash
python app.py
```

---

## 🧪 Usage
1. Open the application  
2. Type/paste one or more news headlines  
3. Click **Predict Sentiment**  
4. View results in a scrollable popup  
5. Save predictions if needed  

---

## 🔮 Future Enhancements
- Export results to CSV  
- Add a GloVe file selector  
- Dark mode UI  
- Convert to `.exe`  
- Integrate live news scraping (Yahoo Finance, Reuters)

---

## 🤝 Author
**Sanya Gupta**  
Machine Learning Model • GUI • NLP Pipeline • Application Logic  
If you like this project, please ⭐ star the repository!


