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
