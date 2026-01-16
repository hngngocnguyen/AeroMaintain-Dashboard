# 🚀 Guide de Déploiement - Dashboard Plotly/Dash
## AeroMaintain Solutions - Maintenance Prédictive

---

## 📋 Vue d'Ensemble

Ce guide fournit les étapes complètes pour transformer les résultats du notebook en un **dashboard interactif en production**.

---

## 🛠️ Architecture

```
┌─────────────────────────────────────────┐
│   Jupyter Notebook (Analyse)            │
│  (Maintenance_Predictive_*.ipynb)       │
└────────────────┬────────────────────────┘
                 │
                 ├─→ predictions_moteurs.csv
                 ├─→ models/ (pickles)
                 └─→ data/ (preprocessed)
                 │
                 ▼
┌─────────────────────────────────────────┐
│   Dashboard Dash (Web App)              │
│  (dashboard_aeromaintain.py)            │
│  - Tabs: Executive / Flotte / RUL / Real-time
│  - Interactif, filtres, exports
└────────────────┬────────────────────────┘
                 │
                 ├─→ Port 8050 (dev)
                 └─→ Production (Gunicorn)
                 │
                 ▼
┌─────────────────────────────────────────┐
│   Utilisateurs Finaux                   │
│  Navigateur Web (localhost:8050)         │
└─────────────────────────────────────────┘
```

---

## 📦 Installation

### Étape 1: Créer l'environnement virtuel

```bash
# Windows PowerShell
python -m venv venv_aero
.\venv_aero\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv_aero
source venv_aero/bin/activate
```

### Étape 2: Installer les dépendances

```bash
pip install --upgrade pip

# Dépendances principales
pip install dash plotly pandas numpy scikit-learn xgboost

# Dépendances de production
pip install gunicorn python-dotenv

# Optionnel: pour export images
pip install kaleido plotly-orca
```

Ou utiliser `requirements.txt`:

```bash
# Créer requirements.txt
pip freeze > requirements.txt

# Installer depuis requirements.txt
pip install -r requirements.txt
```

---

## 💻 Fichiers du Dashboard

### Structure de dossier

```
aeromaintain_dashboard/
├── dashboard_aeromaintain.py      # Application principale
├── requirements.txt               # Dépendances
├── .env                          # Variables d'environnement
├── data/
│   ├── predictions_moteurs.csv   # Données prédictions
│   └── features_data.pkl         # Features preprocessing
├── models/
│   ├── model_rf.pkl              # Modèle Random Forest
│   ├── model_gb.pkl              # Modèle Gradient Boosting
│   └── scaler.pkl                # StandardScaler
├── assets/
│   └── style.css                 # Styling personnalisé
└── logs/
    └── app.log                   # Logs applicatifs
```

---

## 🎨 Code du Dashboard (dashboard_aeromaintain.py)

```python
# ============================================================================
# Dashboard Interactif - AeroMaintain Solutions
# ============================================================================

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Configuration
load_dotenv()
DEBUG = os.getenv('DEBUG', 'False') == 'True'
HOST = os.getenv('DASH_HOST', '127.0.0.1')
PORT = int(os.getenv('DASH_PORT', 8050))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

logger.info("Chargement des données...")

# Charger prédictions
df_predictions = pd.read_csv('data/predictions_moteurs.csv')

# Charger features (optionnel)
try:
    with open('data/features_data.pkl', 'rb') as f:
        df_features = pickle.load(f)
except FileNotFoundError:
    logger.warning("features_data.pkl non trouvé")
    df_features = df_predictions

# Charger modèles
try:
    with open('models/model_rf.pkl', 'rb') as f:
        model_rf = pickle.load(f)
except FileNotFoundError:
    logger.warning("Modèles non chargés")
    model_rf = None

# Palette de couleurs
COLOR_PALETTE = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'neutral': '#95a5a6',
}

logger.info("✅ Données chargées avec succès")

# ============================================================================
# CRÉATION DE L'APPLICATION DASH
# ============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=['https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap']
)

app.title = "AeroMaintain - Maintenance Prédictive"

# ============================================================================
# LAYOUT PRINCIPAL
# ============================================================================

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1('🛩️ AeroMaintain Dashboard', style={'color': 'white', 'margin': 0}),
            html.P('Maintenance Prédictive Intelligente', style={'color': '#ecf0f1', 'margin': 0})
        ], style={'backgroundColor': COLOR_PALETTE['primary'], 'padding': '20px', 'borderRadius': '5px'})
    ], style={'marginBottom': '30px'}),
    
    # Tabs
    dcc.Tabs(id='main-tabs', value='tab-1', children=[
        # ====================================================================
        # ONGLET 1: VUE EXECUTIVE
        # ====================================================================
        dcc.Tab(label='📊 Executive', value='tab-1', children=[
            html.Div([
                html.H2('Vue d\'Ensemble - KPIs Clés', style={'color': COLOR_PALETTE['primary']}),
                
                # KPI Cards
                html.Div([
                    html.Div([
                        html.H3(f"{(df_predictions['risk_level'] == '🔴 Critique').sum()}", 
                               style={'color': COLOR_PALETTE['danger']}),
                        html.P('Moteurs Critiques')
                    ], className='kpi-card'),
                    
                    html.Div([
                        html.H3(f"{(df_predictions['risk_level'] == '🟡 Dégradé').sum()}", 
                               style={'color': COLOR_PALETTE['warning']}),
                        html.P('Moteurs Dégradés')
                    ], className='kpi-card'),
                    
                    html.Div([
                        html.H3(f"{(df_predictions['risk_level'] == '🟢 Sain').sum()}", 
                               style={'color': COLOR_PALETTE['secondary']}),
                        html.P('Moteurs Sains')
                    ], className='kpi-card'),
                    
                    html.Div([
                        html.H3(f"{df_predictions['unit_id'].nunique()}", 
                               style={'color': COLOR_PALETTE['neutral']}),
                        html.P('Total Moteurs')
                    ], className='kpi-card'),
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(4, 1fr)',
                    'gap': '20px',
                    'marginBottom': '30px'
                }),
                
                # Graphiques
                html.Div([
                    html.Div([
                        dcc.Graph(id='risk-distribution-pie')
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(id='rul-histogram')
                    ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
                ]),
                
                html.Div([
                    dcc.Graph(id='critical-engines-table')
                ]),
                
            ], style={'padding': '20px'})
        ]),
        
        # ====================================================================
        # ONGLET 2: ANALYSE DE FLOTTE
        # ====================================================================
        dcc.Tab(label='🔀 Flotte', value='tab-2', children=[
            html.Div([
                html.H2('Segmentation de Flotte', style={'color': COLOR_PALETTE['primary']}),
                
                html.Div([
                    html.Label('Filtre Cluster:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='cluster-filter',
                        options=[
                            {'label': 'Tous les clusters', 'value': -1},
                            *[{'label': f'Cluster {i}', 'value': i} 
                              for i in sorted(df_predictions.get('cluster', []).unique()) 
                              if pd.notna(i)]
                        ],
                        value=-1,
                        multi=False,
                        style={'width': '100%'}
                    ),
                ], style={'marginBottom': '20px', 'width': '300px'}),
                
                dcc.Graph(id='cluster-scatter'),
                dcc.Graph(id='cluster-heatmap'),
                dcc.Graph(id='rul-by-cluster'),
                
            ], style={'padding': '20px'})
        ]),
        
        # ====================================================================
        # ONGLET 3: PRÉDICTIONS
        # ====================================================================
        dcc.Tab(label='🎯 Prédictions', value='tab-3', children=[
            html.Div([
                html.H2('Prédictions RUL par Moteur', style={'color': COLOR_PALETTE['primary']}),
                
                html.Div([
                    html.Label('Sélectionner un moteur:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='engine-selector',
                        options=[
                            {'label': f'Moteur {uid}', 'value': uid} 
                            for uid in sorted(df_predictions['unit_id'].unique())
                        ],
                        value=df_predictions['unit_id'].iloc[0] if len(df_predictions) > 0 else None,
                        multi=False,
                        style={'width': '100%'}
                    ),
                ], style={'marginBottom': '20px', 'width': '300px'}),
                
                html.Div(id='engine-info', style={
                    'backgroundColor': '#ecf0f1',
                    'padding': '15px',
                    'borderRadius': '5px',
                    'marginBottom': '20px'
                }),
                
                dcc.Graph(id='rul-prediction-chart'),
                dcc.Graph(id='sensor-readings'),
                
            ], style={'padding': '20px'})
        ]),
        
        # ====================================================================
        # ONGLET 4: MONITORING
        # ====================================================================
        dcc.Tab(label='🔴 Monitoring', value='tab-4', children=[
            html.Div([
                html.H2('Monitoring & Anomalies', style={'color': COLOR_PALETTE['primary']}),
                
                dcc.Interval(id='refresh-interval', interval=30000, n_intervals=0),
                
                html.Div([
                    html.Div(id='last-update', style={'color': COLOR_PALETTE['neutral']})
                ], style={'marginBottom': '20px'}),
                
                dcc.Graph(id='anomalies-timeline'),
                
                html.H3('Alertes Récentes'),
                html.Table(
                    id='alerts-table',
                    children=[],
                    style={'width': '100%', 'borderCollapse': 'collapse'}
                ),
                
            ], style={'padding': '20px'})
        ]),
        
    ], style={'marginTop': '20px'}),
    
    # Footer
    html.Hr(),
    html.Div([
        html.P(
            '© 2025 AeroMaintain Solutions | Maintenance Prédictive Intelligente',
            style={'textAlign': 'center', 'color': COLOR_PALETTE['neutral'], 'marginTop': '20px'}
        )
    ]),
    
], style={'fontFamily': 'Roboto, sans-serif', 'padding': '20px', 'maxWidth': '1400px', 'margin': '0 auto'})

# ============================================================================
# CALLBACKS INTERACTIFS
# ============================================================================

@callback(
    Output('risk-distribution-pie', 'figure'),
    Input('main-tabs', 'value')
)
def update_risk_distribution(_):
    """Graphique pie distribution du risque"""
    if 'risk_level' not in df_predictions.columns:
        return go.Figure()
    
    risk_counts = df_predictions['risk_level'].value_counts()
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title='Distribution des Moteurs par Niveau de Risque',
        color_discrete_sequence=[COLOR_PALETTE['secondary'], COLOR_PALETTE['warning'], COLOR_PALETTE['danger']]
    )
    return fig

@callback(
    Output('rul-histogram', 'figure'),
    Input('main-tabs', 'value')
)
def update_rul_histogram(_):
    """Histogramme distribution RUL"""
    fig = px.histogram(
        df_predictions,
        x='rul_predicted',
        nbinsx=30,
        title='Distribution du RUL Prédit',
        labels={'rul_predicted': 'RUL (cycles)', 'count': 'Nombre de moteurs'},
        color_discrete_sequence=[COLOR_PALETTE['primary']]
    )
    return fig

@callback(
    Output('cluster-scatter', 'figure'),
    Input('cluster-filter', 'value')
)
def update_cluster_scatter(selected_cluster):
    """Scatter plot des clusters"""
    if 'cluster' not in df_predictions.columns:
        return go.Figure().add_annotation(text="Données de clustering non disponibles")
    
    df_filtered = df_predictions
    if selected_cluster != -1:
        df_filtered = df_predictions[df_predictions['cluster'] == selected_cluster]
    
    fig = px.scatter(
        df_filtered,
        x='rul',
        y='rul_predicted',
        color='risk_level' if 'risk_level' in df_filtered.columns else None,
        hover_data=['unit_id', 'scenario'],
        title=f'RUL Réel vs Prédit{"" if selected_cluster == -1 else f" - Cluster {selected_cluster}"}',
        labels={'rul': 'RUL Réel', 'rul_predicted': 'RUL Prédit'},
        color_discrete_map={
            '🟢 Sain': COLOR_PALETTE['secondary'],
            '🟡 Dégradé': COLOR_PALETTE['warning'],
            '🔴 Critique': COLOR_PALETTE['danger']
        }
    )
    return fig

@callback(
    Output('engine-info', 'children'),
    Input('engine-selector', 'value')
)
def update_engine_info(selected_engine):
    """Afficher les infos du moteur sélectionné"""
    if selected_engine is None:
        return "Sélectionnez un moteur"
    
    df_engine = df_predictions[df_predictions['unit_id'] == selected_engine]
    if len(df_engine) == 0:
        return "Aucune donnée pour ce moteur"
    
    engine_data = df_engine.iloc[0]
    
    return html.Div([
        html.H4(f"Moteur {selected_engine}"),
        html.P(f"Scénario: {engine_data.get('scenario', 'N/A')}"),
        html.P(f"RUL Réel: {engine_data.get('rul', 'N/A'):.0f} cycles"),
        html.P(f"RUL Prédit: {engine_data.get('rul_predicted', 'N/A'):.0f} cycles"),
        html.P(f"Risque: {engine_data.get('risk_level', 'N/A')}"),
    ])

@callback(
    Output('last-update', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def update_timestamp(_):
    """Afficher l'heure de dernière mise à jour"""
    return f"Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}"

# ============================================================================
# LANCER L'APPLICATION
# ============================================================================

if __name__ == '__main__':
    logger.info(f"Démarrage du dashboard sur {HOST}:{PORT}")
    app.run_server(
        debug=DEBUG,
        host=HOST,
        port=PORT,
        dev_tools_ui=DEBUG
    )
    logger.info("Dashboard arrêté")
```

---

## 🌐 Configuration Production

### .env (fichier de configuration)

```bash
# Environment
FLASK_ENV=production
DEBUG=False

# Dashboard
DASH_HOST=0.0.0.0
DASH_PORT=8050

# Database (optionnel)
DATABASE_URL=postgresql://user:password@localhost/aero_db

# API (optionnel)
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here
```

### requirements.txt

```
dash==2.14.2
plotly==5.18.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.0
gunicorn==21.2.0
python-dotenv==1.0.0
```

### Procfile (pour Heroku)

```
web: gunicorn --workers 4 --worker-class sync --bind 0.0.0.0:$PORT --access-logfile - --error-logfile - dashboard_aeromaintain:app.server
```

---

## 🚀 Lancer le Dashboard

### Mode Développement

```bash
# Lancer l'app
python dashboard_aeromaintain.py

# Accéder à: http://localhost:8050
```

### Mode Production (Gunicorn)

```bash
# Lancer avec Gunicorn (4 workers)
gunicorn --workers 4 --bind 0.0.0.0:8050 dashboard_aeromaintain:app.server

# Ou avec superviseur pour auto-restart
supervisord -c /etc/supervisord.conf
```

---

## 📊 Aperçu des Onglets

### Onglet 1: Executive (Vue d'Ensemble)
- KPI cards: Moteurs critiques, dégradés, sains
- Pie chart: Distribution des risques
- Histogramme: RUL distribution
- Table: Top moteurs critiques

### Onglet 2: Flotte (Clustering)
- Scatter: RUL réel vs prédit
- Heatmap: Profils capteurs par cluster
- Box plot: RUL par cluster
- Filtres interactifs

### Onglet 3: Prédictions (RUL)
- Info moteur: Détails sélectionnés
- Line chart: Courbes RUL
- Timeline: Planning maintenance

### Onglet 4: Monitoring (Temps Réel)
- Auto-refresh toutes les 30s
- Timeline des anomalies
- Table historique alertes

---

## 🔐 Sécurité

### Authentification (optionnel)

```python
# Ajouter dans dashboard_aeromaintain.py
import dash_auth

VALID_USERNAME_PASSWORD_PAIRS = {
    'admin': 'password123',
    'user': 'pass456'
}

auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)
```

### HTTPS (optionnel)

```bash
# Générer certificats auto-signés
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Utiliser avec Gunicorn
gunicorn --certfile=cert.pem --keyfile=key.pem --bind 0.0.0.0:443 dashboard_aeromaintain:app.server
```

---

## 📈 Améliorations Futures

- [ ] Ajouter base de données PostgreSQL pour historique
- [ ] Implémenter API REST pour intégration IoT
- [ ] Ajouter authentification utilisateurs
- [ ] Créer alertes email automatiques
- [ ] Intégrer WebSocket pour real-time updates
- [ ] Ajouter export PDF/Excel des rapports

---

## 📞 Support

Email: `data-science@aeromaintain.fr`  
Documentation: `/docs`

**Bon déploiement! 🚀**
