# 🧠 HVAC Pattern Recognition & Anomaly Detection

> **A Machine Learning pipeline for identifying behavioral patterns and anomalies in HVAC energy usage.**

This project implements a robust endpoint for analyzing HVAC sensor data. It utilizes a combination of **Temporal Encoding** (using a Keras-based autoencoder) and **Behavioral Clustering** (using KMeans) to categorize operational states and detect anomalies in real-time.

---

## 🚀 Key Features

*   **Temporal Learning**: Uses a deep learning autoencoder to learn complex time-dependent representations of sensor data.
*   **Behavioral Clustering**: Categorizes HVAC operation into distinct patterns (e.g., "Morning Startup", "Night Setback").
*   **Context-Aware Labels**: Automatically tags patterns with human-readable contexts (e.g., "Weekday Morning", "Weekend Night").
*   **Scalable Deployment**: seamless integration with **AWS SageMaker** for production-grade inference.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `inference.py` | Core inference logic. Handles data preprocessing, feature engineering, and model prediction. |
| `requirements.txt` | List of Python dependencies required for the project. |
| `models/` | Directory containing pretrained artifacts (`temporal_encoder.keras`, `clustering_model.pkl`, `scaler.pkl`). |

---

## 🛠️ Getting Started

### Prerequisites
*   Python 3.10+

### Installation

1.  **Clone or download the repository.**
2.  **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

## 💻 Usage

### Local Inference
You can run the model locally on a CSV file to generate predictions without deploying to AWS.

**Command:**
```bash
python inference.py <path_to_data.csv> [model_directory]
```

**Example:**
```bash
python inference.py test_24h.csv models
```
*This will output the predicted patterns and cluster IDs for the provided data.*

**Sample Output:**

| timestamp           |   cluster_id | pattern_name   |   hour |   weekday |   month | context                     |
|:--------------------|-------------:|:---------------|-------:|----------:|--------:|:----------------------------|
| 2023-01-01 23:50:00 |            0 | Weekend Idle   |     23 |         6 |       1 | Sun Night (Weekend)         |
| 2023-01-02 00:50:00 |            0 | Weekend Idle   |      0 |         0 |       1 | Mon Night (Weekday)         |
| 2023-01-02 01:50:00 |            0 | Weekend Idle   |      1 |         0 |       1 | Mon Night (Weekday)         |
| 2023-01-02 02:50:00 |            0 | Weekend Idle   |      2 |         0 |       1 | Mon Night (Weekday)         |
| 2023-01-02 03:50:00 |            0 | Weekend Idle   |      3 |         0 |       1 | Mon Night (Weekday)         |
| 2023-01-02 04:50:00 |            0 | Weekend Idle   |      4 |         0 |       1 | Mon Night (Weekday)         |
| 2023-01-02 05:50:00 |            0 | Weekend Idle   |      5 |         0 |       1 | Mon Early Morning (Weekday) |
| 2023-01-02 06:50:00 |            0 | Weekend Idle   |      6 |         0 |       1 | Mon Early Morning (Weekday) |
| 2023-01-02 07:50:00 |            0 | Weekend Idle   |      7 |         0 |       1 | Mon Early Morning (Weekday) |
| 2023-01-02 08:50:00 |            0 | Weekend Idle   |      8 |         0 |       1 | Mon Early Morning (Weekday) |
| 2023-01-02 09:50:00 |            0 | Weekend Idle   |      9 |         0 |       1 | Mon Morning (Weekday)       |
| 2023-01-02 10:50:00 |            3 | Office Peak    |     10 |         0 |       1 | Mon Morning (Weekday)       |
| 2023-01-02 11:50:00 |            3 | Office Peak    |     11 |         0 |       1 | Mon Morning (Weekday)       |
| 2023-01-02 12:50:00 |            3 | Office Peak    |     12 |         0 |       1 | Mon Midday (Weekday)        |
| 2023-01-02 13:50:00 |            3 | Office Peak    |     13 |         0 |       1 | Mon Midday (Weekday)        |
| 2023-01-02 14:50:00 |            3 | Office Peak    |     14 |         0 |       1 | Mon Afternoon (Weekday)     |
| 2023-01-02 15:50:00 |            3 | Office Peak    |     15 |         0 |       1 | Mon Afternoon (Weekday)     |
| 2023-01-02 16:50:00 |            3 | Office Peak    |     16 |         0 |       1 | Mon Afternoon (Weekday)     |
| 2023-01-02 17:50:00 |            3 | Office Peak    |     17 |         0 |       1 | Mon Afternoon (Weekday)     |
| 2023-01-02 18:50:00 |            3 | Office Peak    |     18 |         0 |       1 | Mon Evening (Weekday)       |
| 2023-01-02 19:50:00 |            4 | Winter Heating |     19 |         0 |       1 | Mon Evening (Weekday)       |
| 2023-01-02 20:50:00 |            4 | Winter Heating |     20 |         0 |       1 | Mon Evening (Weekday)       |
| 2023-01-02 21:50:00 |            4 | Winter Heating |     21 |         0 |       1 | Mon Evening (Weekday)       |
| 2023-01-02 22:50:00 |            4 | Winter Heating |     22 |         0 |       1 | Mon Night (Weekday)         |
| 2023-01-02 23:50:00 |            4 | Winter Heating |     23 |         0 |       1 | Mon Night (Weekday)         |
| 2023-01-03 00:50:00 |            4 | Winter Heating |      0 |         1 |       1 | Tue Night (Weekday)         |
| 2023-01-03 01:50:00 |            4 | Winter Heating |      1 |         1 |       1 | Tue Night (Weekday)         |


---

## 📊 Feature Inputs
The model expects a CSV input with the following sensor columns:

*   **Temperature**: `T1`, `T2`, `T3`, `T_out`
*   **Humidity**: `RH_1`, `RH_2`, `RH_3`, `RH_out`
*   **Weather**: `Press_mm_hg`, `Windspeed`, `Visibility`, `Tdewpoint`
*   **Energy**: `HVAC_Energy`
*   **Time**: `date` (Timestamp)

---

## ❓ Troubleshooting

> **Inference Errors?**
> *   Ensure your input CSV has at least 144 rows (24 hours of data) as the model uses a sliding window sequence length of 144 steps.


