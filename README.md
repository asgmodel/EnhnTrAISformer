# **README: Implementing EnhnTrAISformer with PyTorch**  

## **🚢 EnhnTrAISformer: Enhancing AIS Data Quality and Ship Trajectory Prediction Using Transformers**  

### **📌 Overview**  
EnhnTrAISformer is an advanced model that combines:  
- **MVRNNAnomalyQuality** for **regression-based anomaly detection and data quality assessment**.  
- **TrAISformer**, a **Transformer-based model** for **AIS (Automatic Identification System) ship trajectory prediction**.  
- Inspired by **minGPT** for a lightweight Transformer implementation.  

### **🚀 Features**  
✔️ **Assess AIS data quality** and detect trajectory anomalies.  
✔️ **Accurate ship trajectory prediction** using Transformer-based architecture.  
✔️ **Handling missing values** and noise reduction in geolocation data.  
✔️ **Optimized self-learning algorithm** to predict future vessel movements.  

---

## **🔧 Requirements**  

Before running the model, ensure you have the necessary dependencies installed:  

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib
```

---

## **📂 Project Structure**  
```
📦 EnhnTrAISformer
├── 📂 data or mydata              # AIS trajectory dataset
│   ├── train.pkl        # Training data
│   ├── valid.pkl        # Validation data
│   └── test.pkl         # Test data
├── 📂 models             # Model scripts
│   ├── EnhnTrAISformer.py   # Transformer implementation
│   ├── MVRNNAnomalyQuality.py       # MVRNNAnomalyQuality for anomaly detection
│   └── main_training.py         # Training script
├── 📂 results            # Saved models and outputs
│   └── model.pt         # Trained model
└── README.md            # This file
```

---

## **🛠️ How to Run the Model**  

### **1️⃣ Data Preparation**  
- The dataset is loaded from `pkl` files.  
- **MVRNNAnomalyQuality** is applied to detect anomalies and enhance data quality.  

### **2️⃣ Train the Model**  
Run the following command to start training:  
```bash
python main_training.py
```
- `main_training.py` loads data, applies anomaly detection, and trains **TrAISformer**.  

### **3️⃣ Test the Model**  
After training, test the model on new data using:  
```bash
python test.py
```

---

## **📊 Evaluation & Results**  


---

## **📜 References**  
- TrAISformer: [https://arxiv.org/abs/2109.03958](https://arxiv.org/abs/2109.03958)  
- minGPT: [https://github.com/karpathy/minGPT](https://github.com/karpathy/minGPT)  

---

## **📩 Contact**  

