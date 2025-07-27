# Fraud-Detection
# 💳 AI-Powered Fraud Detection System

## 🧠 Overview

This project is an **AI-powered fraud detection system** developed for **HSBC Hackathon 2025**. It detects fraudulent transactions using a hybrid ensemble of:

- ✅ **Autoencoder (Deep Learning)**
- ✅ **Isolation Forest (Anomaly Detection)**

The model is deployed using **Streamlit** for a clean, interactive frontend.

---

## 🎯 Goals

- Accurately identify fraudulent transactions.
- Provide a user-friendly interface for testing.
- Reduce false positives using **AND-based ensemble logic**.
- Enable model reusability and deployment via saved model files.

---

## 📁 Dataset

**File:** `Dataset.csv`

| Column         | Description                                                    |
|----------------|----------------------------------------------------------------|
| `step`         | Time step of the transaction                                   |
| `customer`     | Customer identifier                                            |
| `age`          | Age group (as categorical string like '1', '2', ..., 'Unknown')|
| `gender`       | Gender ('M', 'F', 'U', or 'E')                                 |
| `zipcodeOri`   | ZIP code of origin (mostly constant)                          |
| `merchant`     | Merchant identifier                                            |
| `zipMerchant`  | ZIP code of merchant                                           |
| `category`     | Transaction category (e.g., `es_leisure`, `es_transportation`)|
| `amount`       | Transaction amount                                             |
| `fraud`        | Target label (0 = normal, 1 = fraud)                           |

---

## 🧠 Model Architecture

### Autoencoder (Deep Learning)
- Trained **only on non-fraud** data
- Reconstructs input and calculates **Mean Squared Error (MSE)**
- Transactions with high reconstruction error (above 95th percentile) are flagged as fraud

### Isolation Forest
- Trained on the **entire dataset**
- Detects anomalies using decision function
- Flags outliers as potential frauds

---

## 🔗 Ensemble Logic

**AND Ensemble Rule:**

A transaction is labeled as **fraud** if both models agree:

