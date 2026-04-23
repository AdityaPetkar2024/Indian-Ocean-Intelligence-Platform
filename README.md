# 🌊 Indian Ocean Intelligence Platform

A large-scale ocean analytics system that integrates ARGO float data, satellite observations, and fisheries datasets to generate actionable insights across the Indian Ocean.

---

## 🚀 Overview

This project processes **14.5M+ ocean measurements** from ARGO floats and combines them with **Copernicus satellite data (SST, Chlorophyll, SSH)** and **IOTC tuna catch records** to enable intelligent exploration of ocean conditions.

It provides:

* Geospatial ocean analysis
* Climate and anomaly insights
* Fisheries intelligence
* Natural language querying using LLMs

---

## ⚙️ Key Features

* 📊 **Large-scale Data Processing**
  Ingests and processes millions of ocean measurements into PostgreSQL

* 🌍 **Geospatial Intelligence**
  Query ocean conditions by location, region, and time

* 🤖 **LLM-powered Query System**
  Natural language → database queries using LLaMA 3.3 (Groq API)

* 📈 **Ocean Feature Engineering**
  Mixed Layer Depth, Thermocline Depth, Water Mass Classification

* 🗺️ **Interactive Dashboard**
  Streamlit-based UI with maps, T-S diagrams, SST heatmaps, and more

---

## 🌍 Data Coverage

* ARGO float data (**currently INCOIS floats**)
* Copernicus Marine satellite data (SST, Chlorophyll, SSH)
* IOTC tuna catch dataset (67K+ records, multi-decade coverage)

---

## 🧠 Architecture

```text
Raw Data Sources
(ARGO + Copernicus + IOTC)
            ↓
     Data Processing (Python, Xarray)
            ↓
        PostgreSQL Database
            ↓
   Streamlit Dashboard + LLM Interface
```

---

## 🛠️ Tech Stack

* **Python** (Pandas, NumPy, Xarray)
* **PostgreSQL**
* **Streamlit**
* **Scikit-learn (KMeans)**
* **Groq API (LLaMA 3.3)**
* **Geopy (Nominatim)**

---

## 📊 Capabilities

* Ocean anomaly detection
* Regional warming trend analysis
* Float tracking and trajectory visualization
* Tuna catch vs ocean condition analysis
* Water mass clustering using T-S diagrams

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/AdityaPetkar2024/Indian-Ocean-Intelligence-Platform.git
cd Indian-Ocean-Intelligence-Platform
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run salineadd.py
```

---

## ⚠️ Notes

* This repo does **not include large datasets** (.nc, CSVs)
* Requires a PostgreSQL database with preprocessed data
* Designed for scalable deployment (Render / AWS / cloud DBs)

---

## 🔭 Roadmap

* 🌐 Expand beyond INCOIS to **global ARGO coverage**
* 🐟 Integrate additional **fisheries datasets and species-level data**
* 📊 Develop **fish concentration prediction models** using ocean features
* 🧠 Improve ocean feature engineering and anomaly detection
* ⚡ Optimize pipelines for real-time and large-scale processing

---

## 💡 Product Vision

This project is evolving into a **data-driven ocean intelligence platform (SaaS)** for:

* Oceanographic researchers
* Fisheries companies
* Climate and environmental analysts

Planned capabilities:

* 🔌 **API Access** for programmatic queries
* 📍 **Location-based ocean intelligence & predictions**
* ⚡ **Real-time analytics and insights**
* 🐟 **Fishing zone prediction using ML models**

---

## 📌 Future Improvements

* Backend API (FastAPI) for scalable access
* Cloud-native deployment architecture
* Data pipeline optimization and automation
* Integration with additional ocean and climate datasets

---

## 👨‍💻 Author

Built as a high-performance ocean intelligence system combining data engineering, machine learning, and geospatial analytics.

