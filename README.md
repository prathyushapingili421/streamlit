

# 📌 **Customer Segmentation using K-Means & Streamlit**

*A Machine Learning Mini-Project by Prathyusha Pingili*

---

## 📝 **Overview**

This project performs **customer segmentation** using the **K-Means clustering algorithm**.
We simulate customer data with the features:

* **Annual Income (k$)**
* **Spending Score (1–100)**

A machine learning model is trained to group customers into **five clusters**.
A **Streamlit web application** is then created to:

* Accept user input
* Predict which cluster the user belongs to
* Visualize the segmentation using a scatter plot

This project demonstrates fundamental ML concepts such as:

* Data simulation
* Unsupervised learning
* Model persistence (`pickle`)
* Interactive visualization
* Deployment via Streamlit

---

## 📂 **Project Structure**

```
📁 customer-segmentation
│
├── customer_app.py                     # Streamlit application
├── train_customer_model_prathyusha.py  # Script to generate data & train KMeans
│
├── customer_kmeans_model_prathyusha.pkl  # Saved trained model
├── customer_metadata.pkl                 # Feature names + simulated data
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ **Installation & Setup**

### **1. Create a Virtual Environment**

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

---

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

If you created the environment manually, install packages with:

```bash
pip install numpy pandas scikit-learn streamlit matplotlib
```

---

## 🧪 **Model Training**

Run the training script:

```bash
python train_customer_model_prathyusha.py
```

The script will:

✔ Generate 200 simulated data points.

✔ Train a **5-cluster KMeans model.**

✔ Save the trained model to `customer_kmeans_model_prathyusha.pkl`

✔ Save metadata (feature names + original data) to `customer_metadata.pkl`

---

## 🚀 **Running the Streamlit App**

Start the application with:

```bash
streamlit run customer_app.py
```

This will open an interactive webpage in your browser.

---

## 🎛️ **App Features**

### **Sidebar Inputs**

* Annual Income (slider)
* Spending Score (slider)

### **Prediction Output**

Displayed using:

```
Cluster ID → 0, 1, 2, 3, or 4
```

### **Scatter Plot Visualization**

The app displays:

* All simulated data (gray dots)
* K-Means cluster centers (red X)
* The user's input (highlighted in blue)

This allows clear visual interpretation of where the new customer falls.

---

## 📸 **Expected Output Example**

Your submission requires a screenshot of:

✔ Sidebar sliders.

✔ Predicted cluster ID.

✔ Scatter plot with customer point highlighted

*(Insert screenshot in this section once captured.)*

---

## 🧰 **Technologies Used**

* **Python 3**
* **NumPy**
* **Pandas**
* **Scikit-Learn**
* **Streamlit**
* **Matplotlib**

---

## ✨ **Author**

**Prathyusha Pingili**

📍 San Jose, California

GitHub: *[https://github.com/prathyushapingili421](https://github.com/prathyushapingili421)*


