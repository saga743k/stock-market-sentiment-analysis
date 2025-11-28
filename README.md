# 📈 Stock News Sentiment Analyzer

A desktop application that predicts the sentiment (Positive/Negative) of stock market news headlines using a **Deep Learning LSTM model with GloVe embeddings**.  
The model is integrated into a full-screen **Tkinter GUI** and supports **multiple headlines** at once.

---

## 🚀 Features

- 🖥️ Full-screen Tkinter desktop app  
- 📝 Enter one or multiple headlines (one per line)  
- 🔤 GloVe embeddings + LSTM deep learning model  
- 🤖 Real-time sentiment prediction  
- 📊 Confidence score and raw probability  
- 📜 Scrollable results window  
- 💾 Save predictions to a `.txt` file  
- ⚠️ Error handling & clean UI  

---

## 🧠 Machine Learning Model

This project uses a **Deep Learning architecture**:

- **Tokenizer + Padding**  
- **Pre-trained GloVe embeddings** (100d/200d)
- **LSTM layer** for sequence understanding  
- **Dense + Dropout** for classification  
- **Sigmoid output** (Logistic Regression) for binary sentiment  

### 🏗️ Model Architecture
Input (30 tokens)
↓
Embedding (GloVe pretrained vectors)
↓
LSTM (64 units)
↓
Dense (ReLU) + Dropout
↓
Sigmoid Output Layer (0 to 1 score)


---

## 🛠️ Tech Stack

**Language:** Python  
**Libraries:** TensorFlow, Keras, NumPy, NLTK, Tkinter  
**Environment:** VS Code + Virtual Environment (venv)  

---

## 📂 Project Structure



SentimentApp/
│── app.py
│── sentiment_model.keras
│── tokenizer.pkl
│── requirements.txt
│── README.md


---

## 📦 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2️⃣ Create a virtual environment
Windows
python -m venv venv
venv\Scripts\activate

macOS / Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

5️⃣ Run the app
python app.py

🧪 Usage

Open the application

Type/paste one or more news headlines

Click Predict Sentiment

View predictions in a separate results window

Save results to a text file if needed

🔮 Future Enhancements

Export predictions to CSV

Add GloVe file selector

Add dark mode

Convert to .exe

Live news scraping (Yahoo Finance / Reuters)

🤝 Author

Developed by Sanya Gupta
Model, GUI, NLP preprocessing, and application logic.
