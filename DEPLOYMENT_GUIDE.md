# 🚀 Guide de Déploiement du Dashboard Dash

## Installation et Lancement Rapide

### Étape 1 : Installations des Dépendances
```bash
pip install dash plotly pandas numpy scikit-learn
```

### Étape 2 : Créer le Fichier `app.py`

Copier le code ci-dessous dans un fichier `app.py` :

```python
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Charger les données exportées depuis le notebook
df_predictions = pd.read_csv('predictions_moteurs_dashboard.csv')

# Initialiser l'app Dash
app = dash.Dash(__name__)

# ============================================================================
# ONGLET 1 : VUE EXECUTIVE
# ============================================================================

tab1_content = html.Div([
    html.H2('📊 Vue Executive - KPIs Clés', style={'textAlign': 'center', 'marginBottom': 30}),
    
    html.Div([
        html.Div([
            html.H4('Moteurs Sains', style={'color': '#2ecc71'}),
            html.H2(f"{(df_predictions['risk_level'] == '🟢 Sain').sum()}", style={'color': '#2ecc71'})
        ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'flex': 1}),
        
        html.Div([
            html.H4('Dégradés', style={'color': '#f39c12'}),
            html.H2(f"{(df_predictions['risk_level'] == '🟡 Dégradé').sum()}", style={'color': '#f39c12'})
        ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'flex': 1, 'margin': '0 20px'}),
        
        html.Div([
            html.H4('Critiques', style={'color': '#e74c3c'}),
            html.H2(f"{(df_predictions['risk_level'] == '🔴 Critique').sum()}", style={'color': '#e74c3c'})
        ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'flex': 1}),
    ], style={'display': 'flex', 'marginBottom': '30px'}),
    
    dcc.Graph(id='pie-risk-distribution'),
    dcc.Graph(id='bar-critical-engines'),
])

# ============================================================================
# ONGLET 2 : ANALYSE DE FLOTTE
# ============================================================================

tab2_content = html.Div([
    html.H2('🔀 Analyse de Flotte - Clustering', style={'textAlign': 'center', 'marginBottom': 30}),
    
    html.Div([
        html.Label('Filtrer par Cluster:'),
        dcc.Dropdown(
            id='cluster-filter',
            options=[{'label': f'Cluster {c}', 'value': c} for c in sorted(df_predictions['cluster'].unique())],
            value=0,
            style={'width': '100%'}
        ),
    ], style={'marginBottom': '20px'}),
    
    dcc.Graph(id='cluster-distribution'),
])

# ============================================================================
# ONGLET 3 : PRÉDICTIONS
# ============================================================================

tab3_content = html.Div([
    html.H2('🎯 Prédictions & Maintenance', style={'textAlign': 'center', 'marginBottom': 30}),
    
    html.Div([
        html.Label('Sélectionner un moteur:'),
        dcc.Dropdown(
            id='engine-selector',
            options=[{'label': f'Moteur {u}', 'value': u} for u in sorted(df_predictions['unit_id'].unique())],
            value=df_predictions['unit_id'].iloc[0],
            style={'width': '100%'}
        ),
    ], style={'marginBottom': '20px'}),
    
    dcc.Graph(id='rul-prediction-graph'),
    
    html.Div([
        html.H4('Recommandation Maintenance:'),
        html.Div(id='maintenance-recommendation', style={'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px'})
    ])
])

# ============================================================================
# ONGLET 4 : MONITORING
# ============================================================================

tab4_content = html.Div([
    html.H2('🔴 Monitoring Temps Réel', style={'textAlign': 'center', 'marginBottom': 30}),
    
    dcc.Graph(id='heatmap-sensors'),
    
    html.Div([
        html.H4('Historique Alertes:'),
        html.Table([
            html.Thead(
                html.Tr([
                    html.Th('Moteur'),
                    html.Th('RUL Prédit'),
                    html.Th('Niveau Risque'),
                    html.Th('Action Recommandée')
                ])
            ),
            html.Tbody(
                [html.Tr([
                    html.Td(row['unit_id']),
                    html.Td(f"{row['rul_predicted']:.1f}"),
                    html.Td(row['risk_level']),
                    html.Td('Maintenance immédiate' if row['risk_level'] == '🔴 Critique' else 
                           'Surveiller' if row['risk_level'] == '🟡 Dégradé' else 'OK')
                ]) for _, row in df_predictions.nlargest(10, 'rul_error').iterrows()]
            )
        ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '20px'})
    ])
])

# ============================================================================
# LAYOUT PRINCIPAL
# ============================================================================

app.layout = html.Div([
    html.Div([
        html.H1('🛩️ AeroMaintain - Dashboard Maintenance Prédictive', 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        html.Hr()
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'marginBottom': '30px'}),
    
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='📊 Executive', value='tab-1', children=tab1_content, style={'padding': '20px'}),
        dcc.Tab(label='🔀 Flotte', value='tab-2', children=tab2_content, style={'padding': '20px'}),
        dcc.Tab(label='🎯 Prédictions', value='tab-3', children=tab3_content, style={'padding': '20px'}),
        dcc.Tab(label='🔴 Monitoring', value='tab-4', children=tab4_content, style={'padding': '20px'}),
    ]),
    
    html.Div(style={'height': '50px'})  # Espacer du bas
])

# ============================================================================
# CALLBACKS INTERACTIFS
# ============================================================================

@app.callback(
    Output('pie-risk-distribution', 'figure'),
    Input('tabs', 'value')
)
def update_pie(tab):
    risk_counts = df_predictions['risk_level'].value_counts()
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title='Distribution des Niveaux de Risque',
        color_discrete_map={'🟢 Sain': '#2ecc71', '🟡 Dégradé': '#f39c12', '🔴 Critique': '#e74c3c'}
    )
    return fig

@app.callback(
    Output('bar-critical-engines', 'figure'),
    Input('tabs', 'value')
)
def update_critical_engines(tab):
    top_critical = df_predictions.nlargest(10, 'rul_error')
    fig = px.bar(
        top_critical,
        x='rul_error',
        y='unit_id',
        orientation='h',
        title='Top 10 Moteurs à Surveiller (Erreur RUL)',
        labels={'rul_error': 'Erreur RUL (cycles)', 'unit_id': 'Moteur'},
        color='risk_level',
        color_discrete_map={'🟢 Sain': '#2ecc71', '🟡 Dégradé': '#f39c12', '🔴 Critique': '#e74c3c'}
    )
    return fig

@app.callback(
    Output('cluster-distribution', 'figure'),
    Input('cluster-filter', 'value')
)
def update_cluster(selected_cluster):
    df_filtered = df_predictions[df_predictions['cluster'] == selected_cluster]
    fig = px.scatter(
        df_filtered,
        x='rul',
        y='rul_predicted',
        color='risk_level',
        hover_data=['unit_id'],
        title=f'Cluster {selected_cluster} - RUL Réel vs Prédit',
        labels={'rul': 'RUL Réel', 'rul_predicted': 'RUL Prédit'},
        color_discrete_map={'🟢 Sain': '#2ecc71', '🟡 Dégradé': '#f39c12', '🔴 Critique': '#e74c3c'}
    )
    return fig

@app.callback(
    Output('rul-prediction-graph', 'figure'),
    Input('engine-selector', 'value')
)
def update_rul_prediction(selected_engine):
    df_engine = df_predictions[df_predictions['unit_id'] == selected_engine]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_engine['cycles'],
        y=df_engine['rul'],
        mode='lines+markers',
        name='RUL Réel',
        line=dict(color='#2ecc71', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=df_engine['cycles'],
        y=df_engine['rul_predicted'],
        mode='lines+markers',
        name='RUL Prédit',
        line=dict(color='#3498db', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title=f'Évolution RUL - Moteur {selected_engine}',
        xaxis_title='Cycles de Vol',
        yaxis_title='RUL (cycles)',
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

@app.callback(
    Output('maintenance-recommendation', 'children'),
    Input('engine-selector', 'value')
)
def update_maintenance_rec(selected_engine):
    row = df_predictions[df_predictions['unit_id'] == selected_engine].iloc[0]
    
    if row['risk_level'] == '🔴 Critique':
        return html.Div([
            html.H5('⚠️ MAINTENANCE IMMÉDIATE REQUISE', style={'color': '#e74c3c'}),
            html.P(f"RUL estimé: {row['rul_predicted']:.0f} cycles"),
            html.P("Action: Programmer maintenance corrective urgente")
        ], style={'color': '#e74c3c', 'fontWeight': 'bold'})
    elif row['risk_level'] == '🟡 Dégradé':
        return html.Div([
            html.H5('⚠️ MAINTENANCE PRÉVENTIVE À PLANIFIER', style={'color': '#f39c12'}),
            html.P(f"RUL estimé: {row['rul_predicted']:.0f} cycles"),
            html.P("Action: Programmer maintenance préventive dans les 2-3 prochains mois")
        ], style={'color': '#f39c12'})
    else:
        return html.Div([
            html.H5('✅ MOTEUR EN BON ÉTAT', style={'color': '#2ecc71'}),
            html.P(f"RUL estimé: {row['rul_predicted']:.0f} cycles"),
            html.P("Action: Surveiller régulièrement")
        ], style={'color': '#2ecc71'})

@app.callback(
    Output('heatmap-sensors', 'figure'),
    Input('tabs', 'value')
)
def update_heatmap(tab):
    # Utiliser les clusters comme groupes pour heatmap
    cluster_profiles = df_predictions.groupby('cluster').agg({
        'rul_predicted': 'mean',
        'rul_error': 'mean',
        'unit_id': 'count'
    }).reset_index()
    
    fig = px.bar(
        cluster_profiles,
        x='cluster',
        y=['rul_predicted', 'rul_error'],
        title='Profils Moyens par Cluster',
        labels={'cluster': 'Cluster', 'value': 'Valeur'},
        barmode='group'
    )
    return fig

# ============================================================================
# LANCER L'APP
# ============================================================================

if __name__ == '__main__':
    print("🚀 Lancement du Dashboard...")
    print("📍 Accéder à: http://localhost:8050")
    app.run_server(debug=False, host='0.0.0.0', port=8050)
```

### Étape 3 : Lancer le Dashboard
```bash
python app.py
```

### Étape 4 : Accéder au Dashboard
Ouvrir votre navigateur et aller à :
```
http://localhost:8050
```

---

## 📊 Fonctionnalités du Dashboard

✅ **Onglet 1 (Executive)** : Vue synthétique avec KPIs
✅ **Onglet 2 (Flotte)** : Analyse clustering interactif
✅ **Onglet 3 (Prédictions)** : Sélection moteur et évolution RUL
✅ **Onglet 4 (Monitoring)** : Heatmaps et historique alertes

---

## 🔧 Déploiement en Production

### Option 1 : Heroku (Gratuit avec limitations)
```bash
# Installer Heroku CLI
heroku login
heroku create aeromaintain-dashboard
git push heroku main
```

### Option 2 : AWS EC2 / GCP Compute Engine
```bash
# Sur serveur Linux
git clone <repo>
pip install -r requirements.txt
gunicorn app:server -b 0.0.0.0:8050
```

### Option 3 : Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

---

## 📝 Fichier requirements.txt
```
dash==2.14.1
plotly==5.17.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
gunicorn==21.2.0
```

---

## ✅ Troubleshooting

**Port déjà utilisé ?**
```python
app.run_server(port=8051)  # Utiliser un autre port
```

**Données pas trouvées ?**
```python
# Vérifier que predictions_moteurs_dashboard.csv existe dans le même dossier
import os
print(os.path.exists('predictions_moteurs_dashboard.csv'))
```

**ModuleNotFoundError ?**
```bash
pip install --upgrade dash plotly pandas numpy scikit-learn
```

---

**🎉 Dashboard Prêt pour Production !**

