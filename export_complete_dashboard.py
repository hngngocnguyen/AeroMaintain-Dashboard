"""
🛩️ AeroMaintain Dashboard - Complete HTML Export
All 5 dashboards in one beautiful interactive HTML file
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
COLOR_PALETTE = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'danger': '#e74c3c',
    'warning': '#f39c12'
}

RUL_THRESHOLD_CRITICAL = 10
RUL_THRESHOLD_WARNING = 30
FLEET_SIZE = 150

# ==================== LOAD DATA ====================
print("Loading data...")
train_cols = ['unit_id', 'cycles', 'setting1', 'setting2', 'setting3'] + [f'S{i+1}' for i in range(21)]

data_train = pd.read_csv('dataset/train_FD001.txt', sep=r'\s+', names=train_cols, engine='python')
data_test = pd.read_csv('dataset/test_FD001.txt', sep=r'\s+', names=train_cols, engine='python')
rul_test = pd.read_csv('dataset/RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'], engine='python')

def add_engine_targets(data):
    data['cycles_max'] = data.groupby('unit_id')['cycles'].transform('max')
    data['rul_true'] = data['cycles_max'] - data['cycles']
    data['progress_rel'] = data['cycles'] / data['cycles_max']
    return data

data_train = add_engine_targets(data_train)

sensor_cols = [f'S{i+1}' for i in range(21)]

def train_rul_model(data):
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

print("Training RUL model...")
model, feat_cols, eval_metrics = train_rul_model(data_train)

# ==================== GENERATE ALL CHARTS ====================
print("Generating Dashboard Executive...")

# === DASHBOARD EXECUTIVE ===
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

risk_counts = pd.Series(levels).value_counts()

# Chart 1: Risk Pie
fig_exec_pie = px.pie(values=risk_counts.values, names=risk_counts.index, hole=0.4,
                      color_discrete_map={'Critique': COLOR_PALETTE['danger'], 
                                         'Alerte': COLOR_PALETTE['warning'], 
                                         'Normal': COLOR_PALETTE['secondary']})
fig_exec_pie.update_layout(height=400)

# Chart 2: Lifecycle Evolution
df_levels = data_train[['unit_id', 'cycles', 'cycles_max', 'progress_rel']].copy()
df_levels['risk'] = levels
df_levels['bin'] = pd.cut(df_levels['progress_rel'], bins=np.linspace(0, 1, 21), 
                         labels=[f"{int(b*5)}%" for b in range(20)])
evo = df_levels.groupby(['bin', 'risk']).size().reset_index(name='count')
evo_pivot = evo.pivot(index='bin', columns='risk', values='count').fillna(0).reset_index()

fig_exec_area = px.area(evo_pivot, x='bin', y=['Normal', 'Alerte', 'Critique'],
                        color_discrete_map={'Critique': COLOR_PALETTE['danger'], 
                                           'Alerte': COLOR_PALETTE['warning'], 
                                           'Normal': COLOR_PALETTE['secondary']})
fig_exec_area.update_layout(height=400, xaxis_title="Progression", yaxis_title="Moteurs")

print("Generating Dashboard Opérationnel...")

# === DASHBOARD OPERATIONNEL ===
data_train['risk'] = levels

# Chart 3: Heatmap Sensors
heatmap_data = data_train.groupby('risk')[sensor_cols].mean()
fig_oper_heatmap = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, 
                                             y=heatmap_data.index, colorscale='RdYlGn_r'))
fig_oper_heatmap.update_layout(height=400)

# Chart 4: Correlation
corr_matrix = data_train[sensor_cols[:10]].corr()
fig_oper_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, 
                                          y=corr_matrix.columns, colorscale='RdBu', zmid=0))
fig_oper_corr.update_layout(height=400)

# Chart 5: Anomalies Z-Score
s11_data = data_train['S11'].values
z_scores = np.abs((s11_data - np.mean(s11_data)) / np.std(s11_data))
anomalies = z_scores > 3
anomaly_pct = (anomalies.sum() / len(anomalies)) * 100

fig_oper_zscore = px.histogram(z_scores, nbins=50, color_discrete_sequence=[COLOR_PALETTE['primary']])
fig_oper_zscore.add_vline(x=3, line_dash="dash", line_color=COLOR_PALETTE['danger'])
fig_oper_zscore.update_layout(height=400, xaxis_title="Z-Score", yaxis_title="Fréquence")

print("Generating Dashboard Maintenance Prédictive...")

# === DASHBOARD MAINTENANCE PREDICTIVE ===
selected_motor = sorted(data_train['unit_id'].unique())[0]
df_motor = data_train[data_train['unit_id'] == selected_motor].sort_values('cycles').copy()
df_motor['rul_predicted'] = model.predict(df_motor[feat_cols].values)

residuals = df_motor['rul_predicted'] - df_motor['rul_true']
ci = 1.96 * np.std(residuals)

# Chart 6: RUL Prediction with CI
fig_maint_rul = go.Figure()
fig_maint_rul.add_trace(go.Scatter(x=df_motor['cycles'], y=df_motor['rul_predicted'] + ci, 
                                   mode='lines', line=dict(width=0), showlegend=False))
fig_maint_rul.add_trace(go.Scatter(x=df_motor['cycles'], y=df_motor['rul_predicted'] - ci, 
                                   fill='tonexty', mode='lines',
                                   fillcolor='rgba(52, 152, 219, 0.2)', line=dict(width=0), 
                                   name='IC 95%'))
fig_maint_rul.add_trace(go.Scatter(x=df_motor['cycles'], y=df_motor['rul_predicted'], 
                                   mode='lines+markers', name='RUL Prédite',
                                   line=dict(color=COLOR_PALETTE['primary'], width=3)))
fig_maint_rul.add_trace(go.Scatter(x=df_motor['cycles'], y=df_motor['rul_true'], 
                                   mode='lines', name='RUL Réelle',
                                   line=dict(color=COLOR_PALETTE['secondary'], dash='dash', width=2)))
fig_maint_rul.add_hline(y=RUL_THRESHOLD_CRITICAL, line_dash="dot", 
                       line_color=COLOR_PALETTE['danger'], annotation_text="Critique")
fig_maint_rul.add_hline(y=RUL_THRESHOLD_WARNING, line_dash="dot", 
                       line_color=COLOR_PALETTE['warning'], annotation_text="Alerte")
fig_maint_rul.update_layout(height=400, xaxis_title="Cycles", yaxis_title="RUL", hovermode='x unified')

# Chart 7: Feature Engineering - Sensor with rolling average
sensor_choice = 'S11'
window = 15
df_fe = df_motor.copy()
df_fe[f'{sensor_choice}_roll'] = df_fe[sensor_choice].rolling(window=window, min_periods=1).mean()

fig_maint_fe = go.Figure()
fig_maint_fe.add_trace(go.Scatter(x=df_fe['cycles'], y=df_fe[sensor_choice], mode='lines', 
                                  name=f"{sensor_choice} (brut)",
                                  line=dict(color=COLOR_PALETTE['primary'])))
fig_maint_fe.add_trace(go.Scatter(x=df_fe['cycles'], y=df_fe[f'{sensor_choice}_roll'], 
                                  mode='lines', name=f"{sensor_choice} (lissé)",
                                  line=dict(color=COLOR_PALETTE['secondary'])))
fig_maint_fe.update_layout(height=400, xaxis_title="Cycles", yaxis_title="Valeur")

print("Generating Dashboard Analyse & Insights...")

# === DASHBOARD ANALYSE & INSIGHTS ===

# Chart 8: Sensor Variability
sensor_stats = data_train[sensor_cols].describe().T
fig_analyse_box = px.box(sensor_stats, y='std', color_discrete_sequence=[COLOR_PALETTE['primary']])
fig_analyse_box.update_layout(height=400, yaxis_title="Écart-type", showlegend=False)

# Chart 9: Cycles Distribution
cycles_per_unit = data_train.groupby('unit_id')['cycles'].max()
fig_analyse_hist = px.histogram(cycles_per_unit, nbins=30, color_discrete_sequence=[COLOR_PALETTE['primary']])
fig_analyse_hist.update_layout(height=400, xaxis_title="Cycles", yaxis_title="Nb moteurs", showlegend=False)

# Chart 10: Clustering PCA
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
df_vis = pd.DataFrame({'unit_id': agg_df['unit_id'], 'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 
                      'cluster': clusters})

fig_analyse_pca = px.scatter(df_vis, x='PC1', y='PC2', color=df_vis['cluster'].astype(str),
                            color_discrete_sequence=px.colors.qualitative.Set2)
fig_analyse_pca.update_layout(height=400)

# Chart 11: Anomalies Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomalies_if = iso_forest.fit_predict(data_train[sensor_cols])
n_anomalies = (anomalies_if == -1).sum()

fig_analyse_anom = go.Figure(data=[
    go.Bar(name='Normaux', x=['Résultat'], y=[(anomalies_if == 1).sum()], marker_color=COLOR_PALETTE['secondary']),
    go.Bar(name='Anomalies', x=['Résultat'], y=[n_anomalies], marker_color=COLOR_PALETTE['danger'])
])
fig_analyse_anom.update_layout(height=400, barmode='stack')

# Chart 12: Operational Settings
settings_cols = ['setting1', 'setting2', 'setting3']
fig_analyse_settings = make_subplots(rows=1, cols=3, subplot_titles=settings_cols)
for i, col in enumerate(settings_cols, 1):
    fig_analyse_settings.add_trace(go.Histogram(x=data_train[col], nbinsx=30, 
                                               marker_color=COLOR_PALETTE['primary']), row=1, col=i)
fig_analyse_settings.update_layout(height=350, showlegend=False)

# ==================== BUILD HTML ====================
print("Building complete HTML file...")

html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛩️ AeroMaintain Dashboard Complet</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 100%;
            background: white;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.95;
        }}
        
        .nav-tabs {{
            display: flex;
            background: white;
            border-bottom: 2px solid #e9ecef;
            overflow-x: auto;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        
        .nav-tab {{
            padding: 15px 25px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1em;
            font-weight: 500;
            color: #666;
            transition: all 0.3s;
            white-space: nowrap;
        }}
        
        .nav-tab:hover {{
            background: #f8f9fa;
            color: #667eea;
        }}
        
        .nav-tab.active {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
        }}
        
        .dashboard-section {{
            display: none;
            padding: 30px 20px;
            animation: fadeIn 0.5s;
        }}
        
        .dashboard-section.active {{
            display: block;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        .section-title {{
            font-size: 2em;
            color: #667eea;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
            border-left: 4px solid #667eea;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.15);
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
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .chart-title {{
            padding: 15px 20px;
            background: #f8f9fa;
            font-weight: bold;
            font-size: 1.05em;
            color: #333;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .plotly {{
            width: 100% !important;
        }}
        
        .table-container {{
            overflow-x: auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 25px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #e9ecef;
        }}
        
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 30px 20px;
            text-align: center;
            margin-top: 40px;
        }}
        
        .highlight {{
            background: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin-bottom: 20px;
            border-radius: 4px;
        }}
        
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
            
            .nav-tabs {{
                flex-wrap: wrap;
            }}
            
            .nav-tab {{
                flex: 1;
                min-width: 150px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛩️ Tableau de Bord AeroMaintain</h1>
        <p>Maintenance Prédictive pour Moteurs Turbofan avec Machine Learning</p>
    </div>
    
    <div class="nav-tabs">
        <button class="nav-tab active" onclick="showTab('accueil')">🏠 Accueil</button>
        <button class="nav-tab" onclick="showTab('executive')">📊 Dashboard Executive</button>
        <button class="nav-tab" onclick="showTab('operationnel')">🔧 Opérationnel</button>
        <button class="nav-tab" onclick="showTab('maintenance')">🎯 Maintenance Prédictive</button>
        <button class="nav-tab" onclick="showTab('analyse')">📈 Analyse & Insights</button>
    </div>
    
    <!-- ===== ACCUEIL ===== -->
    <div id="accueil" class="dashboard-section active">
        <div class="section-title">🏠 Accueil</div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">📦 Moteurs Totaux</div>
                <div class="metric-value">{FLEET_SIZE}</div>
                <div class="metric-label">NASA C-MAPSS</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">📊 Capteurs</div>
                <div class="metric-value">21</div>
                <div class="metric-label">par moteur</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">📈 Points Données</div>
                <div class="metric-value">~20k</div>
                <div class="metric-label">cycles totaux</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">🤖 Modèle ML</div>
                <div class="metric-value">RF</div>
                <div class="metric-label">RandomForest</div>
            </div>
        </div>
        
        <div style="max-width: 1000px; margin: 40px auto;">
            <h3 style="color: #667eea; margin-bottom: 20px;">✨ Fonctionnalités Clés</h3>
            <ul style="line-height: 2; font-size: 1.05em; list-style: none;">
                <li>✅ <strong>Prédiction RUL</strong> - Estimer la durée de vie restante des moteurs</li>
                <li>✅ <strong>Détection d'Anomalies</strong> - Identifier les défaillances de capteurs</li>
                <li>✅ <strong>Classification des Risques</strong> - Catégoriser par urgence (Critique/Alerte/Normal)</li>
                <li>✅ <strong>Analyse de Clustering</strong> - Segmenter les moteurs par état de santé</li>
                <li>✅ <strong>Surveillance Complète</strong> - Suivre la dégradation en temps réel</li>
            </ul>
            
            <h3 style="color: #667eea; margin: 30px 0 20px 0;">🎯 Avantages</h3>
            <ul style="line-height: 2; font-size: 1.05em; list-style: none;">
                <li>💰 <strong>Économies</strong> - Prévenir les pannes inattendues (ROI: {roi_pct:.0f}%)</li>
                <li>⏰ <strong>Optimisation</strong> - Planifier la maintenance proactivement</li>
                <li>🚀 <strong>Fiabilité</strong> - Maximiser le temps de fonctionnement</li>
                <li>📊 <strong>Décisions Basées sur les Données</strong> - Insights ML actionables</li>
                <li>🛡️ <strong>Système d'Alerte</strong> - Détection précoce des anomalies</li>
            </ul>
        </div>
    </div>
    
    <!-- ===== DASHBOARD EXECUTIVE ===== -->
    <div id="executive" class="dashboard-section">
        <div class="section-title">📊 Dashboard Executive</div>
        
        <div class="highlight">
            <strong>🎯 KPIs Clés:</strong> Flotte: {FLEET_SIZE} | Cycles Moyens: {mean_cycles:.0f} | Coûts Évités: €{savings:,} | ROI: {roi_pct:.0f}%
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">🛩️ Flotte Analysée</div>
                <div class="metric-value">{nb_moteurs}</div>
                <div class="metric-label">moteurs uniques</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">⏱️ Cycles Moyens</div>
                <div class="metric-value">{mean_cycles:.0f}</div>
                <div class="metric-label">par moteur</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">💰 Coûts Évités</div>
                <div class="metric-value">€{savings:,}</div>
                <div class="metric-label">estimés</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">📈 ROI</div>
                <div class="metric-value">{roi_pct:.0f}%</div>
                <div class="metric-label">maintenance préventive</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">🔴 Critique</div>
                <div class="metric-value" style="color: #e74c3c;">{risk_counts.get('Critique', 0)}</div>
                <div class="metric-label">moteurs</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">🟡 Alerte</div>
                <div class="metric-value" style="color: #f39c12;">{risk_counts.get('Alerte', 0)}</div>
                <div class="metric-label">moteurs</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">🟢 Normal</div>
                <div class="metric-value" style="color: #2ecc71;">{risk_counts.get('Normal', 0)}</div>
                <div class="metric-label">moteurs</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">🎯 État de la Flotte</div>
                {fig_exec_pie.to_html(include_plotlyjs=False, div_id="exec_pie")}
            </div>
            
            <div class="chart-container">
                <div class="chart-title">📊 Évolution du Risque (Cycle de Vie)</div>
                {fig_exec_area.to_html(include_plotlyjs=False, div_id="exec_area")}
            </div>
        </div>
    </div>
    
    <!-- ===== DASHBOARD OPERATIONNEL ===== -->
    <div id="operationnel" class="dashboard-section">
        <div class="section-title">🔧 Dashboard Opérationnel</div>
        
        <div class="highlight">
            <strong>🔍 Surveillance:</strong> {int(anomalies.sum())} Anomalies détectées ({anomaly_pct:.2f}% des données)
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">🌡️ Anomalies Z-Score</div>
                <div class="metric-value">{int(anomalies.sum())}</div>
                <div class="metric-label">points</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">📊 % Anomalies</div>
                <div class="metric-value">{anomaly_pct:.2f}%</div>
                <div class="metric-label">du dataset</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">⚙️ Capteurs Moniteurs</div>
                <div class="metric-value">21</div>
                <div class="metric-label">en continu</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">🌡️ Profil Capteurs par Niveau de Risque</div>
                {fig_oper_heatmap.to_html(include_plotlyjs=False, div_id="oper_heatmap")}
            </div>
            
            <div class="chart-container">
                <div class="chart-title">🔗 Corrélation Top 10 Capteurs</div>
                {fig_oper_corr.to_html(include_plotlyjs=False, div_id="oper_corr")}
            </div>
            
            <div class="chart-container">
                <div class="chart-title">🔍 Anomalies Z-Score (S11)</div>
                {fig_oper_zscore.to_html(include_plotlyjs=False, div_id="oper_zscore")}
            </div>
        </div>
    </div>
    
    <!-- ===== DASHBOARD MAINTENANCE PREDICTIVE ===== -->
    <div id="maintenance" class="dashboard-section">
        <div class="section-title">🎯 Dashboard Maintenance Prédictive</div>
        
        <div class="highlight">
            <strong>📊 Modèle ML:</strong> RandomForest | MAE: {eval_metrics['MAE']:.1f} cycles | RMSE: {eval_metrics['RMSE']:.1f} cycles | Dataset: {eval_metrics['test_size']} obs
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">📈 MAE</div>
                <div class="metric-value">{eval_metrics['MAE']:.1f}</div>
                <div class="metric-label">cycles</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">📊 RMSE</div>
                <div class="metric-value">{eval_metrics['RMSE']:.1f}</div>
                <div class="metric-label">cycles</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">🎯 Moteur</div>
                <div class="metric-value">{selected_motor}</div>
                <div class="metric-label">sélectionné</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">🎯 Prédiction RUL - Moteur {selected_motor}</div>
                {fig_maint_rul.to_html(include_plotlyjs=False, div_id="maint_rul")}
            </div>
            
            <div class="chart-container">
                <div class="chart-title">🧩 Feature Engineering - S11</div>
                {fig_maint_fe.to_html(include_plotlyjs=False, div_id="maint_fe")}
            </div>
        </div>
    </div>
    
    <!-- ===== DASHBOARD ANALYSE & INSIGHTS ===== -->
    <div id="analyse" class="dashboard-section">
        <div class="section-title">📈 Dashboard Analyse & Insights</div>
        
        <div class="highlight">
            <strong>📊 Clustering:</strong> {k} clusters | Variance Expliquée: {pca.explained_variance_ratio_.sum()*100:.1f}% | Anomalies: {n_anomalies} détectées
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">🔗 Clusters</div>
                <div class="metric-value">{k}</div>
                <div class="metric-label">groupes</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">📊 Variance PCA</div>
                <div class="metric-value">{pca.explained_variance_ratio_.sum()*100:.1f}%</div>
                <div class="metric-label">expliquée</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">🔍 Anomalies IF</div>
                <div class="metric-value">{n_anomalies}</div>
                <div class="metric-label">détectées</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">📊 Variabilité Capteurs</div>
                {fig_analyse_box.to_html(include_plotlyjs=False, div_id="analyse_box")}
            </div>
            
            <div class="chart-container">
                <div class="chart-title">📈 Distribution Cycles</div>
                {fig_analyse_hist.to_html(include_plotlyjs=False, div_id="analyse_hist")}
            </div>
            
            <div class="chart-container">
                <div class="chart-title">🔗 Clustering (PCA)</div>
                {fig_analyse_pca.to_html(include_plotlyjs=False, div_id="analyse_pca")}
            </div>
            
            <div class="chart-container">
                <div class="chart-title">🔍 Détection Anomalies</div>
                {fig_analyse_anom.to_html(include_plotlyjs=False, div_id="analyse_anom")}
            </div>
            
            <div class="chart-container" style="grid-column: span 2;">
                <div class="chart-title">⚙️ Paramètres Opérationnels</div>
                {fig_analyse_settings.to_html(include_plotlyjs=False, div_id="analyse_settings")}
            </div>
        </div>
    </div>
    
    <div class="footer">
        <h3>🛩️ AeroMaintain Dashboard - Dashboard Complet</h3>
        <p>Généré le {pd.Timestamp.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>
        <p>Dataset NASA C-MAPSS FD001 • Maintenance Prédictive pour Moteurs Turbofan</p>
        <p style="margin-top: 15px; opacity: 0.8;">Tous les graphiques sont interactifs - survolez pour voir les détails, cliquez et zoomez pour explorer</p>
    </div>
    
    <script>
        function showTab(tabName) {{
            // Hide all sections
            const sections = document.querySelectorAll('.dashboard-section');
            sections.forEach(section => section.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.nav-tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected section and mark tab as active
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
            
            // Scroll to top
            window.scrollTo(0, 0);
        }}
    </script>
</body>
</html>
"""

# Save HTML
output_file = 'AeroMaintain_Dashboard_COMPLET.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ Dashboard complet exporté: {output_file}")
print(f"📊 Ouvrez '{output_file}' dans votre navigateur!")
