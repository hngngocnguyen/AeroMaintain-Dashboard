# 🛩️ AeroMaintain Dashboard - Streamlit Application

## 📋 Overview

This is a Streamlit-based interactive dashboard for predictive maintenance of turbofan engines using the NASA C-MAPSS FD001 dataset.

## 🚀 Quick Start

### Installation

1. **Install dependencies:**
```bash
pip install -r streamlit_requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 Dashboard Features

### 1. **🏠 Home**
- Overview of the project and dataset
- Key metrics (100 engines, 21 sensors, ~20,000 cycles)
- Project objectives and benefits

### 2. **📈 Exploratory Analysis**
- **Sensor Statistics**: Summary statistics of all 21 sensors
- **Correlation Analysis**: Heatmap of sensor relationships
- **Operational Settings**: Distribution of control parameters
- **RUL Distribution**: Engine operating cycle distribution

### 3. **🎯 Predictions**
- Interactive RUL prediction for any engine
- Confidence intervals (95% CI) for prediction uncertainty
- Comparison of predicted vs actual RUL
- Visual threshold indicators (Critical: ≤10 cycles, Warning: ≤30 cycles)
- Real-time status indicator

### 4. **🔍 Anomaly Detection**
Three detection methods:
- **Z-Score**: Statistical outlier detection (|Z| > 3)
- **Isolation Forest**: Machine learning-based anomaly detection
- **Rolling Correlation**: Detects sensor degradation patterns

### 5. **📊 Monitoring**
- Real-time fleet health overview
- Risk distribution pie chart (Critical/Warning/Normal)
- Sensor heatmap by risk level
- Maintenance priority visualization

## 📁 Project Structure

```
.
├── app.py                          # Main Streamlit application
├── streamlit_requirements.txt       # Python dependencies
├── AeroMaintain_Dashboard_*.ipynb  # Jupyter notebooks (reference)
├── dataset/
│   ├── train_FD001.txt            # Training data
│   ├── test_FD001.txt             # Test data
│   └── RUL_FD001.txt              # RUL reference values
└── README_STREAMLIT.md            # This file
```

## 🎨 Features

### 🎯 Interactive Components
- **Sidebar Navigation**: Easy page switching
- **Dropdown Selectors**: Choose specific engines to analyze
- **Dynamic Metrics**: Real-time KPI calculations
- **Plotly Charts**: Fully interactive visualizations

### 📊 Visualizations
- Histograms (sensor distribution, operating cycles)
- Heatmaps (correlation matrix, sensor profiles)
- Line charts (RUL predictions, rolling correlations)
- Pie charts (fleet risk distribution)
- Scatter plots (anomaly detection)

### 🎯 Business Intelligence
- Remaining Useful Life (RUL) predictions
- Maintenance scheduling recommendations
- Cost-benefit analysis indicators
- Risk-based prioritization

## 🔧 Configuration

### Thresholds
```python
RUL_THRESHOLD_CRITICAL = 10   # Immediate maintenance needed
RUL_THRESHOLD_WARNING = 30    # Preventive maintenance scheduled
```

### Color Palette
```python
COLOR_PALETTE = {
    'primary': '#3498db',      # Blue
    'secondary': '#2ecc71',    # Green
    'warning': '#f39c12',      # Orange
    'danger': '#e74c3c',       # Red
    'neutral': '#95a5a6',      # Gray
    'dark': '#2c3e50'          # Dark blue
}
```

## 📈 Data Pipeline

1. **Load Data**: Read NASA C-MAPSS FD001 dataset
2. **Preprocessing**: Standardize sensor values
3. **Feature Engineering**: Calculate rolling statistics
4. **Model Training**: RandomForest + XGBoost
5. **Prediction**: RUL estimation with confidence intervals
6. **Visualization**: Render interactive charts

## 🎓 Dataset Information

**Source**: [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

**FD001 Scenario**:
- 100 turbofan engines
- 21 active sensors per engine
- 3 operational settings
- ~20,600 total operating cycles
- Run-to-failure simulation

**Sensors Include**:
- T2, T24, T30 (temperatures)
- P2, P15, P30 (pressures)
- NF, NC (fan/core speeds)
- And 13 others...

## 🔄 Updating the App

To add new features or modify existing ones:

1. Edit `app.py`
2. Save the file
3. Streamlit will auto-reload (if configured)
4. Or press `R` in the Streamlit interface

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
pip install -r streamlit_requirements.txt
```

### Issue: "Data not found"
Ensure the `dataset/` folder exists with:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

### Issue: "Port already in use"
```bash
streamlit run app.py --server.port 8502
```

## 📊 Performance Tips

- The app uses `@st.cache_data` to cache data loading
- First run will load data, subsequent runs use cache
- Refresh cache: `Ctrl+C` and restart, or add `?clear_cache=true` to URL

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Deploy directly from repository

### Docker Deployment
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r streamlit_requirements.txt
CMD ["streamlit", "run", "app.py"]
```

## 📝 Notes

- This app works with the NASA C-MAPSS FD001 dataset
- All visualizations are interactive (hover, zoom, pan)
- Confidence intervals assume normal distribution of residuals
- Risk levels are calculated using sensor standard deviations

## 🤝 Contributing

To improve the dashboard:
1. Add new analysis methods
2. Implement advanced ML models
3. Create custom metrics
4. Enhance visualizations

## 📞 Support

For issues or questions:
1. Check the Jupyter notebook: `AeroMaintain_Dashboard_Maintenance_Predictive.ipynb`
2. Review NASA C-MAPSS documentation
3. Consult scikit-learn/Plotly documentation

---

**Built with ❤️ using Streamlit, Plotly, and Scikit-learn**
