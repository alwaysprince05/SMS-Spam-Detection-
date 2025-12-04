# 📱 SMS Spam Detection using Machine Learning

A Machine Learning project to classify SMS messages as **Spam** or **Not Spam (Ham)** using text preprocessing and classification algorithms in Python.

---

## 📌 Project Overview

The goal of this project is to build a model that can automatically detect whether a given SMS message is **spam** or **legitimate** based on its content.

This project includes:

- Data cleaning & preprocessing  
- Text vectorization using TF-IDF  
- Model building (Naive Bayes, etc.)  
- Model evaluation (Accuracy, Precision, Confusion Matrix)  
- A simple web app interface for users to test SMS messages

---

## 🧠 Problem Statement

Spam messages are **unsolicited / unwanted / nuisance** messages that can be annoying and sometimes fraudulent.  
This project aims to **filter such spam** from normal SMS messages using Machine Learning.

---

## 🔧 Tech Stack

- **Language:** Python  
- **Libraries:**  
  - pandas, numpy  
  - scikit-learn  
  - NLTK (for text preprocessing)  
  - streamlit / flask (for web app)  
- **Model(s) Used:**  
  - Multinomial Naive Bayes  
  - (Optional) Linear SVM / Logistic Regression for comparison

---

## 🗂️ Dataset

- **Name:** SMS Spam Collection Dataset  
- **Format:** `.csv` file with columns like:
  - `label` → `ham` or `spam`
  - `message` → SMS text

> Replace this with your actual dataset info:

- **Source:** `UCI Machine Learning Repository` / Kaggle / Provided by faculty  
- **Number of rows:** `XXXX`  
- **Class distribution:**  
  - Ham: `XXXX` messages  
  - Spam: `XXXX` messages  

---

## 🧹 Project Workflow

1. **Data Loading**
   - Read the CSV file using `pandas`.

2. **Data Cleaning**
   - Handle missing values.
   - Remove duplicate messages.

3. **Text Preprocessing**
   - Convert text to lowercase.  
   - Remove punctuation, numbers, and special characters.  
   - Remove stopwords (like *is, the, and*).  
   - Perform stemming / lemmatization.

4. **Feature Extraction**
   - Convert text to numerical features using:
     - **TF-IDF Vectorizer** (Term Frequency – Inverse Document Frequency).

5. **Train–Test Split**
   - Split data into **train** and **test** sets (e.g., 80% train, 20% test).

6. **Model Building**
   - Train **Multinomial Naive Bayes** on the transformed data.
   - (Optional) Train other models (SVM, Logistic Regression) for comparison.

7. **Model Evaluation**
   - Metrics used:
     - **Accuracy Score**
     - **Precision Score** (for spam class)
     - **Confusion Matrix**
   - Choose the best model based on performance.

8. **Web App / Deployment**
   - Build a simple web interface (Streamlit / Flask).
   - Take SMS text as input from user.
   - Show prediction: **Spam / Not Spam**.

---

## 🧮 Why Naive Bayes for This Project?

- Works very well on **text data** (like SMS).  
- Assumes features (words) are **independent**, which makes it mathematically simple and fast.  
- Trains quickly even on **high-dimensional sparse** data (like TF-IDF).  
- Gives **good precision** for spam class in many text-classification tasks.

---

## 📊 Results

> Fill this section with your real numbers.

- **Best Model:** Multinomial Naive Bayes  
- **Accuracy:** `94.12 %`  
- **Precision (Spam class):** `1.00 %`  
- **Confusion Matrix:**
  - True Ham: `956`
  - False Ham: `14`
  - True Spam: `147`
  - False Spam: `3`

You can also add a confusion matrix image or classification report screenshot.

---

## 🖥️ How to Run the Project Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/sms-spam-detection.git
cd sms-spam-detection
