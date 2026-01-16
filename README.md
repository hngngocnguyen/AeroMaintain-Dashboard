# 🛩️ AeroMaintain Dashboard

**Predictive Maintenance for Turbofan Engines using Machine Learning**

An intelligent dashboard for monitoring, analyzing, and predicting the remaining useful life (RUL) of aircraft turbofan engines using NASA C-MAPSS dataset.

## ✨ Features

- 📊 **Executive Dashboard** - High-level KPIs and fleet overview
- 🔧 **Operational Dashboard** - Real-time sensor monitoring and anomaly detection
- 🎯 **Predictive Maintenance** - RUL prediction with confidence intervals
- 📈 **Advanced Analytics** - Clustering, anomaly detection, and feature engineering
- 🏠 **Interactive Accueil** - Project introduction and capabilities

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AeroMaintain-Dashboard.git
cd AeroMaintain-Dashboard

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# Launch Streamlit app
streamlit run app.py --server.port 8503
```

Then open your browser to: **http://localhost:8503**

### Export to HTML

To share as a single HTML file (no installation needed):

```bash
python export_complete_dashboard.py
```

This generates `AeroMaintain_Dashboard_COMPLET.html` - open in any browser!

## 📊 Dashboards

### 📊 Dashboard Executive
- Fleet health status (Critical/Alert/Normal)
- Risk evolution over engine lifecycle
- Financial KPIs and ROI analysis
- Cost savings estimation

### 🔧 Dashboard Opérationnel
- Sensor health heatmaps
- Correlation analysis between sensors
- Z-Score anomaly detection
- Real-time alerts

### 🎯 Dashboard Maintenance Prédictive
- RUL (Remaining Useful Life) predictions
- Model performance metrics (MAE, RMSE)
- Feature engineering visualization
- Maintenance timeline planning

### 📈 Dashboard Analyse & Insights
- Sensor variability analysis
- Engine lifecycle distribution
- PCA-based clustering (2D visualization)
- Isolation Forest anomaly detection
- Operational parameters analysis

## 📁 Project Structure

```
AeroMaintain-Dashboard/
├── app.py                              # Main Streamlit application
├── export_complete_dashboard.py        # HTML export script
├── export_to_html.py                   # Simple HTML export
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore file
├── README.md                           # This file
├── dataset/
│   ├── train_FD001.txt                # Training data (NASA C-MAPSS)
│   ├── test_FD001.txt                 # Test data
│   └── RUL_FD001.txt                  # Ground truth RUL
└── notebooks/                          # Jupyter notebooks (optional)
    └── AeroMaintain_Dashboard_Maintenance_Predictive.ipynb
```

## 🛠️ Technical Stack

- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations
- **scikit-learn** - Machine learning
  - RandomForestRegressor - RUL prediction
  - IsolationForest - Anomaly detection
  - KMeans - Clustering
  - PCA - Dimensionality reduction
- **Pandas & NumPy** - Data processing

## 📊 Model Details

### RUL Prediction Model
- **Algorithm**: Random Forest Regressor (200 trees)
- **Features**: 21 sensor readings
- **Target**: Remaining Useful Life (cycles)
- **Train/Test Split**: 80/20 by engine ID
- **Metrics**:
  - MAE: ~15 cycles
  - RMSE: ~25 cycles

### Anomaly Detection
- **Algorithm**: Isolation Forest (5% contamination)
- **Detection**: Z-Score (threshold: 3σ)
- **Coverage**: All 21 sensors

### Clustering
- **Algorithm**: K-Means (3 clusters)
- **Dimensionality Reduction**: PCA (2 components)
- **Engine Segmentation**: Health-based groups

## 📈 Dataset

**NASA C-MAPSS FD001**
- 100 turbofan engines (simulation)
- 21 sensors per engine
- 3 operational settings
- ~20,000 data points total
- Cycles to failure (ground truth)

## 🎯 Use Cases

✅ **Predictive Maintenance Planning**
- Prevent unexpected failures
- Optimize maintenance scheduling
- Reduce downtime

✅ **Cost Optimization**
- ROI: ~167% on preventive maintenance
- Estimated savings: €2,500 per engine/year

✅ **Fleet Management**
- Real-time health monitoring
- Early warning system
- Data-driven decisions

## 🔧 Configuration

Edit constants in `app.py`:
```python
RUL_THRESHOLD_CRITICAL = 10    # Days to failure (critical alert)
RUL_THRESHOLD_WARNING = 30     # Days to failure (warning)
FLEET_SIZE = 150               # Total fleet size
```

## 📝 How to Update

```bash
# Make changes locally
git add .
git commit -m "Your commit message"
git push origin main
```

## 🤝 Contributing

Feel free to fork, modify, and submit pull requests!

## 📄 License

MIT License - See LICENSE file

## 👤 Author

Created for Maintenance Predictive Analytics project

## 📞 Support

For issues or questions, open an issue on GitHub.

---

**🌟 If you find this useful, please star the repository!**
