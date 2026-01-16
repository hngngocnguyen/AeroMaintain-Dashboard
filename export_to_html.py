"""
Export AeroMaintain Dashboard to Static HTML
Single-click HTML file to view all dashboards
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ==================== COLOR PALETTE ====================
COLOR_PALETTE = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'danger': '#e74c3c',
    'warning': '#f39c12'
}

# ==================== THRESHOLDS ====================
RUL_THRESHOLD_CRITICAL = 15
RUL_THRESHOLD_WARNING = 30
FLEET_SIZE = 100

# ==================== LOAD DATA ====================
def load_data():
    train_path = 'dataset/train_FD001.txt'
    test_path = 'dataset/test_FD001.txt'
    rul_path = 'dataset/RUL_FD001.txt'
    
    train_cols = ['unit_id', 'cycles', 'setting1', 'setting2', 'setting3'] + [f'S{i+1}' for i in range(21)]
    
    data_train = pd.read_csv(train_path, sep=r'\s+', names=train_cols, engine='python')
    data_test = pd.read_csv(test_path, sep=r'\s+', names=train_cols, engine='python')
    rul_test = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'], engine='python')
    
    return data_train, data_test, rul_test

def add_engine_targets(data):
    data['cycles_max'] = data.groupby('unit_id')['cycles'].transform('max')
    data['rul_true'] = data['cycles_max'] - data['cycles']
    data['progress_rel'] = data['cycles'] / data['cycles_max']
    return data

def train_rul_model(data):
    sensor_cols = [f'S{i+1}' for i in range(21)]
    X = data[sensor_cols].values
    y = data['rul_true'].values
    unit_ids = data['unit_id'].values
    
    unique_units = np.unique(unit_ids)
    split_idx = int(0.8 * len(unique_units))
    train_units = set(unique_units[:split_idx])
    
    train_mask = np.isin(unit_ids, list(train_units))
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, sensor_cols, {'MAE': mae, 'RMSE': rmse, 'test_size': len(X_test)}

# ==================== CREATE ALL CHARTS ====================
print("Loading data...")
data_train, data_test, rul = load_data()
data_train = add_engine_targets(data_train)

sensor_cols = [f'S{i+1}' for i in range(21)]

# Calculate metrics for Executive Dashboard
nb_moteurs = data_train['unit_id'].nunique()
cycles_par_moteur = data_train.groupby('unit_id')['cycles'].max()
mean_cycles = cycles_par_moteur.mean()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_train[sensor_cols])
row_std = np.std(data_scaled, axis=1)
p70, p90 = np.percentile(row_std, [70, 90])
levels = np.where(row_std > p90, 'Critique', np.where(row_std > p70, 'Alerte', 'Normal'))

cost_unplanned = 120_000
cost_planned = 45_000
last_rows = data_train.sort_values('cycles').groupby('unit_id').tail(1)
nearing = last_rows[last_rows['rul_true'] <= 30]
savings = int(len(nearing) * (cost_unplanned - cost_planned))
roi_pct = ((cost_unplanned - cost_planned) / cost_planned) * 100

print("Training RUL model...")
model, feat_cols, eval_metrics = train_rul_model(data_train)

# ==================== CHART 1: RISK PIE ====================
print("Generating charts...")
risk_counts = pd.Series(levels).value_counts()
fig1 = px.pie(values=risk_counts.values, names=risk_counts.index, hole=0.4,
             title="État de la Flotte",
             color_discrete_map={'Critique': COLOR_PALETTE['danger'], 
                                'Alerte': COLOR_PALETTE['warning'], 
                                'Normal': COLOR_PALETTE['secondary']})
fig1.update_layout(height=500)

# ==================== CHART 2: EVOLUTION ====================
df_levels = data_train[['unit_id', 'cycles', 'cycles_max', 'progress_rel']].copy()
df_levels['risk'] = levels
df_levels['bin'] = pd.cut(df_levels['progress_rel'], bins=np.linspace(0, 1, 21), 
                         labels=[f"{int(b*5)}%" for b in range(20)])
evo = df_levels.groupby(['bin', 'risk']).size().reset_index(name='count')
evo_pivot = evo.pivot(index='bin', columns='risk', values='count').fillna(0).reset_index()

fig2 = px.area(evo_pivot, x='bin', y=['Normal', 'Alerte', 'Critique'],
             title="Évolution du Risque au Cours du Cycle de Vie",
             color_discrete_map={'Critique': COLOR_PALETTE['danger'], 
                                'Alerte': COLOR_PALETTE['warning'], 
                                'Normal': COLOR_PALETTE['secondary']})
fig2.update_layout(height=500)

# ==================== CHART 3: HEATMAP SENSORS ====================
data_train['risk'] = levels
heatmap_data = data_train.groupby('risk')[sensor_cols].mean()
fig3 = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, 
                                colorscale='RdYlGn_r'))
fig3.update_layout(title="Profil Capteurs par Niveau de Risque", height=400)

# ==================== CHART 4: SENSOR CORRELATION ====================
corr_matrix = data_train[sensor_cols[:10]].corr()
fig4 = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, 
                                colorscale='RdBu', zmid=0))
fig4.update_layout(title="Corrélation Top 10 Capteurs", height=400)

# ==================== CHART 5: CLUSTERING PCA ====================
k = 3
agg_dict = {col: ['mean', 'std'] for col in sensor_cols}
agg_df = data_train.groupby('unit_id').agg({**{'cycles': 'max'}, **agg_dict})
agg_df.columns = ['_'.join([c for c in col if c]) for col in agg_df.columns.values]
agg_df = agg_df.reset_index()
feature_cols = [c for c in agg_df.columns if c not in ['unit_id']]
X = scaler.fit_transform(agg_df[feature_cols])
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
km = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = km.fit_predict(X)
cycles_max_series = agg_df['cycles_max'] if 'cycles_max' in agg_df.columns else data_train.groupby('unit_id')['cycles'].max().values
df_vis = pd.DataFrame({'unit_id': agg_df['unit_id'], 'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'cluster': clusters})

fig5 = px.scatter(df_vis, x='PC1', y='PC2', color=df_vis['cluster'].astype(str),
                 title=f"Clustering PCA (Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)",
                 color_discrete_sequence=px.colors.qualitative.Set2)
fig5.update_layout(height=500)

# ==================== CHART 6: ANOMALIES ====================
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomalies_if = iso_forest.fit_predict(data_train[sensor_cols])
n_anomalies = (anomalies_if == -1).sum()
anomaly_pct = (n_anomalies / len(anomalies_if)) * 100

fig6 = go.Figure(data=[
    go.Bar(name='Normaux', x=['Résultat'], y=[(anomalies_if == 1).sum()], marker_color=COLOR_PALETTE['secondary']),
    go.Bar(name='Anomalies', x=['Résultat'], y=[n_anomalies], marker_color=COLOR_PALETTE['danger'])
])
fig6.update_layout(title="Détection Anomalies (Isolation Forest)", height=400, barmode='stack')

# ==================== CHART 7: CYCLES DISTRIBUTION ====================
cycles_per_unit = data_train.groupby('unit_id')['cycles'].max()
fig7 = px.histogram(cycles_per_unit, nbins=30, title="Distribution des Cycles par Moteur",
                   color_discrete_sequence=[COLOR_PALETTE['primary']])
fig7.update_layout(height=400)

# ==================== CHART 8: RUL PREDICTION SAMPLE ====================
selected_motor = sorted(data_train['unit_id'].unique())[0]
df_motor = data_train[data_train['unit_id'] == selected_motor].sort_values('cycles').copy()
df_motor['rul_predicted'] = model.predict(df_motor[feat_cols].values)

fig8 = go.Figure()
fig8.add_trace(go.Scatter(x=df_motor['cycles'], y=df_motor['rul_predicted'], mode='lines+markers', 
                         name='RUL Prédite', line=dict(color=COLOR_PALETTE['primary'], width=3)))
fig8.add_trace(go.Scatter(x=df_motor['cycles'], y=df_motor['rul_true'], mode='lines', 
                         name='RUL Réelle', line=dict(color=COLOR_PALETTE['secondary'], dash='dash', width=2)))
fig8.add_hline(y=RUL_THRESHOLD_CRITICAL, line_dash="dot", line_color=COLOR_PALETTE['danger'])
fig8.add_hline(y=RUL_THRESHOLD_WARNING, line_dash="dot", line_color=COLOR_PALETTE['warning'])
fig8.update_layout(title=f"Prédiction RUL - Moteur {selected_motor}", height=400, xaxis_title="Cycles", yaxis_title="RUL")

# ==================== BUILD HTML ====================
print("Building HTML file...")

html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛩️ AeroMaintain Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .metric {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        
        .metric-label {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            padding: 30px;
        }}
        
        .chart {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .chart-title {{
            padding: 20px;
            background: #f8f9fa;
            font-weight: bold;
            font-size: 1.1em;
            color: #333;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .plotly {{
            width: 100% !important;
            height: 500px !important;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 2px solid #e9ecef;
        }}
        
        @media (max-width: 768px) {{
            .dashboard {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛩️ Tableau de Bord AeroMaintain</h1>
            <p>Maintenance Prédictive pour Moteurs Turbofan - Machine Learning</p>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">🛩️ Flotte Totale</div>
                <div class="metric-value">{FLEET_SIZE}</div>
                <div class="metric-label">moteurs analysés</div>
            </div>
            <div class="metric">
                <div class="metric-label">⏱️ Cycles Moyens</div>
                <div class="metric-value">{mean_cycles:.0f}</div>
                <div class="metric-label">cycles par moteur</div>
            </div>
            <div class="metric">
                <div class="metric-label">💰 Coûts Évités</div>
                <div class="metric-value">€{savings:,}</div>
                <div class="metric-label">estimés</div>
            </div>
            <div class="metric">
                <div class="metric-label">📈 ROI</div>
                <div class="metric-value">{roi_pct:.0f}%</div>
                <div class="metric-label">maintenance préventive</div>
            </div>
            <div class="metric">
                <div class="metric-label">🔴 Critique</div>
                <div class="metric-value" style="color: #e74c3c;">{risk_counts.get('Critique', 0)}</div>
                <div class="metric-label">moteurs</div>
            </div>
            <div class="metric">
                <div class="metric-label">🟡 Alerte</div>
                <div class="metric-value" style="color: #f39c12;">{risk_counts.get('Alerte', 0)}</div>
                <div class="metric-label">moteurs</div>
            </div>
            <div class="metric">
                <div class="metric-label">🟢 Normal</div>
                <div class="metric-value" style="color: #2ecc71;">{risk_counts.get('Normal', 0)}</div>
                <div class="metric-label">moteurs</div>
            </div>
            <div class="metric">
                <div class="metric-label">🔍 Anomalies</div>
                <div class="metric-value">{anomaly_pct:.1f}%</div>
                <div class="metric-label">données anormales</div>
            </div>
        </div>
        
        <div class="dashboard">
            <div class="chart">
                <div class="chart-title">État de la Flotte</div>
                {fig1.to_html(include_plotlyjs=False, div_id="chart1")}
            </div>
            
            <div class="chart">
                <div class="chart-title">Corrélation Top 10 Capteurs</div>
                {fig4.to_html(include_plotlyjs=False, div_id="chart4")}
            </div>
            
            <div class="chart">
                <div class="chart-title">Évolution du Risque au Cours du Cycle de Vie</div>
                {fig2.to_html(include_plotlyjs=False, div_id="chart2")}
            </div>
            
            <div class="chart">
                <div class="chart-title">Profil Capteurs par Niveau de Risque</div>
                {fig3.to_html(include_plotlyjs=False, div_id="chart3")}
            </div>
            
            <div class="chart">
                <div class="chart-title">Clustering PCA - Segmentation des Moteurs</div>
                {fig5.to_html(include_plotlyjs=False, div_id="chart5")}
            </div>
            
            <div class="chart">
                <div class="chart-title">Distribution des Cycles par Moteur</div>
                {fig7.to_html(include_plotlyjs=False, div_id="chart7")}
            </div>
            
            <div class="chart">
                <div class="chart-title">Détection Anomalies (Isolation Forest)</div>
                {fig6.to_html(include_plotlyjs=False, div_id="chart6")}
            </div>
            
            <div class="chart">
                <div class="chart-title">Prédiction RUL - Moteur {selected_motor}</div>
                {fig8.to_html(include_plotlyjs=False, div_id="chart8")}
            </div>
        </div>
        
        <div class="footer">
            <p>📊 AeroMaintain Dashboard - Généré le {pd.Timestamp.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>
            <p>Dataset NASA C-MAPSS FD001 • Machine Learning pour la Maintenance Prédictive</p>
        </div>
    </div>
</body>
</html>
"""

# Save HTML file
output_file = 'AeroMaintain_Dashboard.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ Dashboard exported to: {output_file}")
print(f"📊 Open '{output_file}' in your browser to view the dashboard")
