# 🛩️ AeroMaintain Dashboard - Getting Started Guide

## 📚 Complete Setup Instructions

### Option 1: Quick Start (Recommended for Windows)

1. **Double-click `run_streamlit.bat`**
   - Dependencies will install automatically
   - Dashboard opens in browser at `http://localhost:8501`

### Option 2: Manual Setup (All Platforms)

#### Step 1: Install Dependencies
```bash
pip install -r streamlit_requirements.txt
```

#### Step 2: Run the Application
```bash
streamlit run app.py
```

#### Step 3: Access the Dashboard
- Open browser: `http://localhost:8501`
- Dashboard loads with 5 main sections

---

## 🎯 Dashboard Tour

### Section 1: 🏠 Home
- **Purpose**: Project overview and statistics
- **Shows**:
  - Total engines in dataset (100)
  - Number of sensors (21)
  - Data volume (~20,000 cycles)
  - Key features and benefits
  - Dataset information

**Use Case**: Understanding the project scope and objectives

---

### Section 2: 📈 Exploratory Data Analysis

#### 2.1 Sensor Statistics
- Summary table: Mean, Std, Min, Max, Quartiles
- Bar chart of sensor variability
- Identifies most/least variable sensors

**Use Case**: Understanding sensor ranges and distributions

#### 2.2 Correlation Analysis
- Heatmap showing sensor relationships
- Identifies correlated sensor pairs
- Highlights redundant sensors

**Use Case**: Feature engineering and sensor selection

#### 2.3 Operational Settings
- Distribution of 3 control settings (setting1, setting2, setting3)
- Shows operating conditions ranges
- Identifies operational modes

**Use Case**: Understanding flight conditions and their impact

#### 2.4 RUL Distribution
- Histogram of operating cycles per engine
- Shows engine lifespan variability
- Statistical summaries

**Use Case**: Understanding degradation patterns

---

### Section 3: 🎯 Predictions

#### Features:
- **Interactive Selection**: Choose any engine (1-20) from dropdown
- **RUL Trend**: Predicted vs actual remaining useful life
- **Confidence Interval**: 95% uncertainty band (shaded blue area)
- **Visual Thresholds**:
  - 🔴 Red line at RUL ≤ 10 (Critical)
  - 🟡 Orange line at RUL ≤ 30 (Warning)
- **Status Indicator**: Current engine status badge
- **Metrics Dashboard**: 4 key metrics below chart

#### Interpretation:
- **Narrow band**: High prediction confidence
- **Wide band**: High uncertainty (model should be retrained)
- **Crossing thresholds**: Triggers maintenance alerts
- **Diverging lines**: Model drift - requires adjustment

**Use Case**: Predict maintenance needs and schedule interventions

---

### Section 4: 🔍 Anomaly Detection

#### Method 1: Z-Score
- **Formula**: Z = |X - μ| / σ
- **Threshold**: |Z| > 3 (0.3% of normal data)
- **Best for**: Single sensor outliers
- **Output**: Histogram + anomaly count

#### Method 2: Isolation Forest
- **Algorithm**: ML-based isolation-based anomaly detection
- **Contamination**: 5% of data expected as anomalies
- **Best for**: Multivariate anomalies
- **Output**: Anomaly counts and percentages

#### Method 3: Rolling Correlation
- **Monitors**: Relationship between S11 and S12
- **Window**: 30-cycle rolling calculation
- **Thresholds**:
  - -0.75: Warning (correlation degrading)
  - -0.60: Critical (correlation near failure)
- **Best for**: Systemic degradation detection

**Use Case**: Early detection of sensor failures or system degradation

---

### Section 5: 📊 Real-Time Monitoring

#### Features:
- **Fleet Health Pie Chart**: Distribution of engines by risk level
  - 🔴 Critical (red)
  - 🟡 Warning (orange)
  - 🟢 Normal (green)
- **Risk Metrics**: Count of engines in each category
- **Sensor Heatmap**: Average sensor values by risk level
  - Red = High risk sensor values
  - Green = Normal sensor values

#### Risk Calculation:
```
Risk Score = Standard Deviation of Normalized Sensors
- High SD = High variability = Higher risk
- Low SD = Stable = Lower risk
```

**Use Case**: Monitor overall fleet health and identify critical engines

---

## 🔧 Customization

### Change Color Scheme
Edit `app.py` line ~28:
```python
COLOR_PALETTE = {
    'primary': '#3498db',      # Main blue
    'secondary': '#2ecc71',    # Success green
    'warning': '#f39c12',      # Alert orange
    'danger': '#e74c3c',       # Critical red
}
```

### Adjust Thresholds
Edit `app.py` line ~36-37:
```python
RUL_THRESHOLD_CRITICAL = 10   # Immediate action needed
RUL_THRESHOLD_WARNING = 30    # Schedule maintenance
```

### Modify Anomaly Detection
Edit relevant sections in app.py:
- Z-Score threshold (default: 3)
- Isolation Forest contamination (default: 0.05)
- Rolling correlation window (default: 30)

---

## 🚀 Advanced Deployment

### Option A: Docker (Recommended for Production)

```bash
# Build image
docker build -t aeromaintain .

# Run container
docker run -p 8501:8501 aeromaintain

# Or use docker-compose
docker-compose up
```

### Option B: Streamlit Cloud (Free Hosting)

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repository
4. Dashboard deployed automatically!

### Option C: Cloud Platforms

**AWS EC2**:
```bash
git clone <repo>
cd <project>
pip install -r streamlit_requirements.txt
streamlit run app.py
```

**Heroku** (with Procfile):
```
web: streamlit run app.py --logger.level=error
```

**DigitalOcean/Linode**: Similar to EC2 setup

---

## 📊 Data Requirements

Ensure these files exist in `dataset/` folder:
```
dataset/
├── train_FD001.txt      # Training data (100 engines, 21 sensors)
├── test_FD001.txt       # Test data
└── RUL_FD001.txt        # True RUL values
```

**Data Format** (space-separated values):
```
unit_id cycles setting1 setting2 setting3 S1 S2 S3 ... S21
```

---

## 🎓 Tutorial: End-to-End Workflow

### Scenario: Monitor Engine #15

1. **Go to Home tab**
   - Understand dataset: 100 engines, 21 sensors

2. **Go to EDA tab → Sensor Statistics**
   - Learn sensor ranges and normal behavior

3. **Go to EDA tab → Correlation Analysis**
   - Identify key sensor relationships

4. **Go to Predictions tab**
   - Select Engine 15 from dropdown
   - View RUL prediction with confidence interval
   - Check if RUL is below warning threshold

5. **Go to Anomaly Detection tab**
   - Run Z-Score method
   - Check for anomalies in Engine 15 data

6. **Go to Monitoring tab**
   - Check Engine 15's risk level in fleet
   - See sensor profile heatmap

7. **Decision**:
   - RUL < 10 + Anomalies detected → Schedule immediate maintenance
   - 10 < RUL < 30 → Schedule preventive maintenance
   - RUL > 30 → Continue monitoring

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **Import Error** | Run: `pip install -r streamlit_requirements.txt` |
| **Port 8501 in use** | Run: `streamlit run app.py --server.port 8502` |
| **Data not found** | Check `dataset/` folder contains required files |
| **Slow loading** | Clear cache: Ctrl+C and restart, or add `?clear_cache=true` to URL |
| **Chart not rendering** | Ensure Plotly is installed: `pip install plotly` |

---

## 📈 Performance Metrics

**Expected Performance**:
- Page load: < 3 seconds (first run)
- Page load: < 1 second (cached)
- Chart rendering: < 1 second
- Prediction calculation: < 500ms

**Optimization Tips**:
- App uses `@st.cache_data` for 10x speedup
- Only recalculates when data changes
- Browser-side rendering with Plotly

---

## 🔄 Continuous Improvement

To enhance the dashboard:

1. **Add Real-Time Data**:
   - Connect to live sensor feeds
   - Update predictions every N seconds

2. **Implement ML Models**:
   - Train XGBoost/LSTM models
   - Improve RUL prediction accuracy

3. **Database Integration**:
   - Store predictions in SQL/NoSQL
   - Track historical accuracy

4. **Alerts System**:
   - Email/SMS notifications
   - Slack integration for teams

5. **Mobile App**:
   - Responsive design for phones
   - Push notifications

---

## 📞 Getting Help

- **Streamlit Docs**: https://docs.streamlit.io
- **Plotly Docs**: https://plotly.com/python/
- **Scikit-learn Docs**: https://scikit-learn.org/
- **NASA C-MAPSS Info**: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

---

## 📄 Files Created

```
.
├── app.py                          # Main Streamlit app (450+ lines)
├── streamlit_requirements.txt       # Python dependencies
├── README_STREAMLIT.md             # Full documentation
├── GETTING_STARTED.md              # This file
├── run_streamlit.bat               # Windows launcher
├── run_streamlit.sh                # Unix/Mac launcher
├── Dockerfile                      # Docker container config
├── docker-compose.yml              # Docker compose config
└── .streamlit/config.toml          # Streamlit settings
```

---

**Happy analyzing! 🎉**

For questions or improvements, refer to the main Jupyter notebook for detailed implementation.
