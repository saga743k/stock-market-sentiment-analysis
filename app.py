# app.py — multiple headlines version
import tkinter as tk
from tkinter import messagebox, scrolledtext
import traceback
import sys
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Try to import/load tensorflow here and catch errors early
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception as e:
    print("ERROR importing TensorFlow or Keras:", e)
    traceback.print_exc()
    messagebox.showerror("Import Error", f"Error importing TensorFlow/Keras:\n{e}")
    raise

print("Starting app.py")

# -----------------------------
# LOAD MODEL & TOKENIZER
# -----------------------------
MODEL_FILE = "sentiment_model.keras"   # change to .h5 if you used h5
TOKENIZER_FILE = "tokenizer.pkl"

model = None
tokenizer = None

try:
    print("Loading tokenizer from:", TOKENIZER_FILE)
    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded. Example keys (first 20):",
          list(getattr(tokenizer, "word_index", {}).keys())[:20] if tokenizer else "no tokenizer.word_index")
except Exception as e:
    print("ERROR loading tokenizer:", e)
    traceback.print_exc()
    messagebox.showerror("Tokenizer Load Error", f"Failed to load tokenizer.pkl:\n{e}")

try:
    print("Loading model from:", MODEL_FILE, "(this may take a few seconds)")
    model = load_model(MODEL_FILE)
    print("Model loaded successfully.")
    try:
        model.summary()
    except Exception:
        pass
except Exception as e:
    print("ERROR loading model:", e)
    traceback.print_exc()
    messagebox.showerror("Model Load Error", f"Failed to load model:\n{e}")

# -----------------------------
# PREPROCESSING
# -----------------------------
try:
    stop_words = stopwords.words('english')
except Exception:
    print("NLTK stopwords missing; attempting to download...")
    import nltk as _nltk
    _nltk.download('stopwords')
    stop_words = stopwords.words('english')

stemmer = SnowballStemmer('english')
clean_re = r"@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text):
    text = re.sub(clean_re, " ", str(text).lower()).strip()
    tokens = [stemmer.stem(t) for t in text.split() if t not in stop_words]
    return " ".join(tokens)

# -----------------------------
# Utility: determine expected length
# -----------------------------
def get_model_expected_len(fallback=30):
    try:
        expected_len = None
        if model is None:
            return fallback
        if hasattr(model, "input_shape"):
            shape = model.input_shape
            if isinstance(shape, (list, tuple)) and len(shape) >= 2 and shape[1] is not None:
                expected_len = int(shape[1])
        if expected_len is None:
            # try first layer
            try:
                first_layer = model.layers[0]
                if hasattr(first_layer, "input_shape"):
                    s = first_layer.input_shape
                    if isinstance(s, (list, tuple)) and len(s) >= 2 and s[1] is not None:
                        expected_len = int(s[1])
            except Exception:
                pass
        if expected_len is None:
            expected_len = fallback
        return expected_len
    except Exception as e:
        print("Error getting model expected length:", e)
        return fallback

# -----------------------------
# PREDICT — multiple headlines
# -----------------------------
def predict_sentiment():
    try:
        raw_text = text_box.get("1.0", "end-1c")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip() != ""]
        print("Received headlines:", lines)

        if not lines:
            messagebox.showwarning("Empty Input", "Please enter one or more headlines (one per line).")
            return

        if tokenizer is None or model is None:
            messagebox.showerror("Not Ready", "Model or tokenizer not loaded. Check terminal for errors.")
            print("Model or tokenizer is None. tokenizer:", tokenizer, "model:", model)
            return

        preprocessed = [preprocess(h) for h in lines]
        print("Preprocessed headlines:", preprocessed)

        seqs = tokenizer.texts_to_sequences(preprocessed)
        print("Sequences:", seqs)

        maxlen = get_model_expected_len(fallback=30)
        print("Using padding maxlen =", maxlen)
        padded = pad_sequences(seqs, maxlen=maxlen, padding="post")
        print("Padded shape:", padded.shape)

        scores = model.predict(padded)
        print("Raw model outputs:", scores)

        # convert scores to scalars reliably
        scalar_scores = []
        scores_arr = np.asarray(scores)
        if scores_arr.ndim == 1:
            scalar_scores = [float(x) for x in scores_arr]
        elif scores_arr.ndim == 2:
            scalar_scores = [float(x[0]) for x in scores_arr]
        else:
            # fall back flatten
            scalar_scores = [float(np.ravel(x)[0]) for x in scores_arr]

        results = []
        for h, s in zip(lines, scalar_scores):
            sentiment = "Positive" if s > 0.6 else "Negative"
            confidence = s if s > 0.6 else 1 - s
            results.append((h, sentiment, confidence, s))

        # Show results in a new scrollable window
        show_results_window(results)

    except Exception as e:
        print("Exception in predict_sentiment:", e)
        traceback.print_exc()
        messagebox.showerror("Prediction Error", f"An error occurred:\n{e}")

# -----------------------------
# UI: results window
# -----------------------------
# -----------------------------
# UI: results window (FIXED)
# -----------------------------
def show_results_window(results):
    win = tk.Toplevel(window)
    win.title("Prediction Results")
    win.geometry("900x600")

    label = tk.Label(win, text="Predictions", font=("Segoe UI", 20, "bold"))
    label.pack(pady=8)

    st = scrolledtext.ScrolledText(win, width=110, height=30, font=("Segoe UI", 12))
    st.pack(padx=10, pady=10, fill="both", expand=True)

    # header
    st.insert("end", f"{'Headline':<80} | {'Sentiment':<10} | {'Confidence':<9} | {'Raw score':<8}\n")
    st.insert("end", "-" * 120 + "\n")

    for h, sentiment, conf, raw in results:
        # Use proper width + precision formatting:
        # - headline truncated to 75 chars, left-aligned in 80 chars
        # - sentiment left-aligned in 10 chars
        # - confidence displayed with 2 decimal places and width 9
        # - raw score shown with 4 decimal places and width 8
        display_line = f"{h[:75]:<80} | {sentiment:<10} | {conf:9.2f} | {raw:8.4f}\n"
        st.insert("end", display_line)

    st.configure(state="disabled")

    # Add a simple "Save as text" button
    def save_results():
        try:
            with open("prediction_results.txt", "w", encoding="utf-8") as f:
                f.write(st.get("1.0", "end"))
            messagebox.showinfo("Saved", "Results saved to prediction_results.txt")
        except Exception as ex:
            messagebox.showerror("Save Error", f"Could not save results:\n{ex}")

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=6)
    tk.Button(btn_frame, text="Save Results to file", command=save_results, font=("Segoe UI", 12)).pack()

# -----------------------------
# FULL SCREEN UI
# -----------------------------
window = tk.Tk()
window.title("Stock News Sentiment Analyzer")
window.state("zoomed")
window.config(bg="#f0f4f7")

title_label = tk.Label(
    window,
    text="📈 Stock News Sentiment Analyzer",
    font=("Segoe UI", 32, "bold"),
    bg="#f0f4f7",
    fg="#2C3E50"
)
title_label.pack(pady=20)

frame = tk.Frame(window, bg="white", bd=3, relief="solid")
frame.pack(pady=20, padx=50, ipadx=20, ipady=20, fill="x")

input_label = tk.Label(frame, text="Enter News Headlines (one per line):", font=("Segoe UI", 18), bg="white")
input_label.pack(pady=10)

text_box = tk.Text(frame, height=8, width=100, font=("Segoe UI", 14))
text_box.pack(pady=10)

button = tk.Button(
    frame,
    text="Predict Sentiment",
    command=predict_sentiment,
    font=("Segoe UI", 18, "bold"),
    bg="#3498DB",
    fg="white",
    width=24,
    height=1
)
button.pack(pady=12)

result_label = tk.Label(window, text="", font=("Segoe UI", 20, "bold"), bg="#f0f4f7")
result_label.pack(pady=10)

print("GUI constructed, entering mainloop()")
window.mainloop()
print("App exited")
