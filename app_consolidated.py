"""
🛩️ AeroMaintain Dashboard - Comprehensive Multi-Visual Dashboards
Intelligent Predictive Maintenance for Turbofan Engines
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🛩️ Tableau de Bord AeroMaintain",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLING ====================
st.markdown("""
    <style>
    .main {padding-top: 0rem;}
    h1 {color: #3498db; text-align: center; margin-bottom: 1rem;}
    h2 {color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 0.5rem;}
    h3 {color: #2c3e50; margin-top: 1rem;}
    </style>
""", unsafe_allow_html=True)

# ==================== COLOR PALETTE ====================
COLOR_PALETTE = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'neutral': '#95a5a6',
    'dark': '#2c3e50'
}

# ==================== CONSTANTS ====================
RUL_THRESHOLD_CRITICAL = 10
RUL_THRESHOLD_WARNING = 30
FLEET_SIZE = 150

# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    dataset_path = Path('dataset')
    train_data = pd.read_csv(
        dataset_path / 'train_FD001.txt', sep=r'\s+', header=None,
        names=['unit_id', 'cycles', 'setting1', 'setting2', 'setting3'] + [f'S{i+1}' for i in range(21)]
    )
    test_data = pd.read_csv(
        dataset_path / 'test_FD001.txt', sep=r'\s+', header=None,
        names=['unit_id', 'cycles', 'setting1', 'setting2', 'setting3'] + [f'S{i+1}' for i in range(21)]
    )
    rul_data = pd.read_csv(dataset_path / 'RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])
    return train_data, test_data, rul_data

@st.cache_data
def add_engine_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cycles_max = df.groupby('unit_id')['cycles'].transform('max')
    df['cycles_max'] = cycles_max
    df['rul_true'] = df['cycles_max'] - df['cycles']
    df['progress_rel'] = df['cycles'] / (df['cycles_max'] + 1e-8)
    return df

@st.cache_resource
def train_rul_model(df: pd.DataFrame):
    sensor_cols = [f'S{i+1}' for i in range(21)]
    setting_cols = ['setting1', 'setting2', 'setting3']
    feat_cols = setting_cols + sensor_cols
    engines = df['unit_id'].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(engines)
    cut = int(0.8 * len(engines))
    train_engines, test_engines = set(engines[:cut]), set(engines[cut:])
    train_df = df[df['unit_id'].isin(train_engines)]
    test_df = df[df['unit_id'].isin(test_engines)]
    X_train, y_train = train_df[feat_cols].values, train_df['rul_true'].values
    X_test, y_test = test_df[feat_cols].values, test_df['rul_true'].values
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return model, feat_cols, {"MAE": mae, "RMSE": rmse, "test_size": len(y_test)}

# ==================== SIDEBAR ====================
st.sidebar.markdown("# ⚙️ Configuration")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📊 Sélectionnez le Dashboard:",
    [
        "🏠 Accueil",
        "📊 Dashboard Executive",
        "🔧 Dashboard Opérationnel",
        "🎯 Dashboard Maintenance Prédictive",
        "📈 Dashboard Analyse & Insights"
    ],
    index=0
)

st.sidebar.markdown("---")

if page == "🏠 Accueil":
    st.sidebar.markdown("""
**Dataset NASA C-MAPSS FD001**
- ~100 moteurs turbofan
- 21 capteurs par moteur
- Données jusqu'à panne

### 🎯 Objectifs
- Prédire la RUL
- Détecter les anomalies
- Optimiser la maintenance
""")

# ==================== MAIN HEADER ====================
st.markdown("# 🛩️ Tableau de Bord AeroMaintain - Prédiction Intelligente de Maintenance")
st.markdown("**Maintenance Prédictive pour Moteurs Turbofan utilisant le Machine Learning**")
st.markdown("---")

# ==================== PAGE ROUTING ====================

if page == "🏠 Accueil":
    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Moteurs Totaux", FLEET_SIZE, "NASA C-MAPSS FD001")
    col2.metric("📊 Capteurs", 21, "par moteur")
    col3.metric("📈 Points de Données", "~20 000", "cycles totaux")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✨ Fonctionnalités Clés")
        st.write("""
        ✅ **Prédiction RUL** - Durée de vie restante  
        ✅ **Détection d'Anomalies** - Capteurs  
        ✅ **Classification Risques** - Par urgence  
        ✅ **Clustering** - Segmentation santé  
        ✅ **Surveillance** - Temps réel  
        """)
    with col2:
        st.subheader("🎯 Avantages")
        st.write("""
        💰 **Économies** - Prévenir pannes  
        ⏰ **Optimisation** - Planning proactif  
        🚀 **Fiabilité** - Maximiser uptime  
        📊 **Data-Driven** - Insights ML  
        🛡️ **Mitigation Risques** - Alertes  
        """)
    
    st.markdown("---")
    try:
        data_train, data_test, rul = load_data()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Échantillons Train", len(data_train))
        col2.metric("Échantillons Test", len(data_test))
        col3.metric("Moteurs Uniques", data_train['unit_id'].nunique())
        col4.metric("Max Cycles", int(data_train.groupby('unit_id')['cycles'].max().max()))
    except Exception as e:
        st.error(f"⚠️ Erreur chargement : {e}")

elif page == "📊 Dashboard Executive":
    st.subheader("📊 Dashboard Executive — Vue Stratégique Complète")
    
    try:
        data_train, _, _ = load_data()
        data_train = add_engine_targets(data_train)
        sensor_cols = [f'S{i+1}' for i in range(21)]
        
        nb_moteurs = data_train['unit_id'].nunique()
        cycles_par_moteur = data_train.groupby('unit_id')['cycles'].max()
        mean_cycles = cycles_par_moteur.mean()
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_train[sensor_cols])
        row_std = np.std(data_scaled, axis=1)
        p70, p90 = np.percentile(row_std, [70, 90])
        levels = np.where(row_std > p90, 'Critique', np.where(row_std > p70, 'Alerte', 'Normal'))
        
        cost_unplanned, cost_planned = 120_000, 45_000
        last_rows = data_train.sort_values('cycles').groupby('unit_id').tail(1)
        nearing = last_rows[last_rows['rul_true'] <= 30]
        savings = int(len(nearing) * (cost_unplanned - cost_planned))
        roi_pct = ((cost_unplanned - cost_planned) / cost_planned) * 100
        
        # ROW 1: KPIs
        st.markdown("### 💼 Indicateurs Clés")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🛩️ Flotte", FLEET_SIZE)
        c2.metric("⚙️ Analysés", int(nb_moteurs))
        c3.metric("⏱️ Cycles Moy", f"{mean_cycles:.0f}")
        c4.metric("💰 Économies", f"€{savings:,}".replace(","," "))
        c5.metric("📈 ROI", f"{roi_pct:.0f}%")
        
        st.markdown("---")
        
        # ROW 2: PIE + LIFECYCLE
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### 🎯 État Flotte")
            risk_counts = pd.Series(levels).value_counts()
            fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index, hole=0.4,
                            color_discrete_map={'Critique': COLOR_PALETTE['danger'], 
                                               'Alerte': COLOR_PALETTE['warning'], 
                                               'Normal': COLOR_PALETTE['secondary']})
            fig_pie.update_layout(height=350, margin=dict(t=30, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("🔴", int(risk_counts.get('Critique', 0)))
            m2.metric("🟡", int(risk_counts.get('Alerte', 0)))
            m3.metric("🟢", int(risk_counts.get('Normal', 0)))
        
        with col2:
            st.markdown("### 📊 Évolution Risques")
            df_levels = data_train[['progress_rel']].copy()
            df_levels['risk'] = levels
            df_levels['bin'] = pd.cut(df_levels['progress_rel'], bins=np.linspace(0, 1, 21), 
                                     labels=[f"{int(b*5)}%" for b in range(20)])
            evo = df_levels.groupby(['bin', 'risk']).size().reset_index(name='count')
            evo_pivot = evo.pivot(index='bin', columns='risk', values='count').fillna(0).reset_index().sort_values('bin')
            fig_area = px.area(evo_pivot, x='bin', y=['Normal', 'Alerte', 'Critique'],
                              color_discrete_map={'Critique': COLOR_PALETTE['danger'], 
                                                 'Alerte': COLOR_PALETTE['warning'], 
                                                 'Normal': COLOR_PALETTE['secondary']})
            fig_area.update_layout(height=350, xaxis_title="Progression", yaxis_title="Moteurs", margin=dict(t=30, b=50))
            st.plotly_chart(fig_area, use_container_width=True)
        
        st.markdown("---")
        
        # ROW 3: RUL SEGMENTS + TOP RISKS
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔍 RUL par Segment (k=3)")
            agg_dict = {col: ['mean', 'std'] for col in sensor_cols}
            agg_df = data_train.groupby('unit_id').agg({**{'cycles': 'max'}, **agg_dict})
            agg_df.columns = ['_'.join([c for c in col if c]) for col in agg_df.columns.values]
            agg_df = agg_df.reset_index()
            feat_cols = [c for c in agg_df.columns if c not in ['unit_id']]
            X = StandardScaler().fit_transform(agg_df[feat_cols])
            km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels3 = km3.fit_predict(X)
            cluster_map = pd.DataFrame({'unit_id': agg_df['unit_id'], 'cluster': labels3})
            last_rows_c = last_rows.merge(cluster_map, on='unit_id', how='left')
            kpi_rul = last_rows_c.groupby('cluster')['rul_true'].mean().round(1)
            fig_bar = px.bar(x=[f"Seg {i}" for i in range(3)], y=[kpi_rul.get(i, 0) for i in range(3)],
                            color=[f"Seg {i}" for i in range(3)], color_discrete_sequence=px.colors.qualitative.Set2)
            fig_bar.update_layout(height=300, showlegend=False, yaxis_title="RUL (cycles)", margin=dict(t=20, b=30))
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.markdown("### ⚠️ Top 10 Risques")
            risk_by_engine = pd.DataFrame({'unit_id': data_train['unit_id'], 'row_risk': row_std}).groupby('unit_id')['row_risk'].mean().reset_index(name='score')
            top_risk = risk_by_engine.nlargest(10, 'score').rename(columns={'unit_id': 'Moteur', 'score': 'Score'})
            top_risk['Niveau'] = 'Critique'
            st.dataframe(top_risk[['Moteur', 'Score', 'Niveau']], use_container_width=True, height=300)
        
        st.markdown("---")
        
        # ROW 4: CYCLES + ACTIONS
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 📈 Distribution Cycles")
            fig_hist = px.histogram(cycles_par_moteur, nbins=25, color_discrete_sequence=[COLOR_PALETTE['primary']])
            fig_hist.update_layout(height=300, xaxis_title="Cycles max", yaxis_title="Nb moteurs", showlegend=False, margin=dict(t=20, b=50))
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.markdown("### 💡 Actions")
            st.info(f"""
✅ **{int(risk_counts.get('Critique', 0))}** moteurs critiques  
⚠️ **{int(risk_counts.get('Alerte', 0))}** en alerte  
💰 Économies: **€{savings:,}**  
📈 ROI: **{roi_pct:.0f}%**
            """.replace(",", " "))
    
    except Exception as e:
        st.error(f"⚠️ Erreur: {e}")

elif page == "🔧 Dashboard Opérationnel":
    st.subheader("🔧 Dashboard Opérationnel — Surveillance Complète")
    
    try:
        data_train, _, _ = load_data()
        sensor_cols = [f'S{i+1}' for i in range(21)]
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_train[sensor_cols])
        risk_scores = np.std(data_scaled, axis=1)
        risk_levels = ['Critical' if s > np.percentile(risk_scores, 90) else 'Warning' if s > np.percentile(risk_scores, 70) else 'Normal' for s in risk_scores]
        risk_counts = pd.Series(risk_levels).value_counts()
        
        # ROW 1: FLEET + HEATMAP
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### 🛩️ Santé Flotte")
            fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index, hole=0.4,
                            color_discrete_map={'Critical': COLOR_PALETTE['danger'],
                                               'Warning': COLOR_PALETTE['warning'],
                                               'Normal': COLOR_PALETTE['secondary']})
            fig_pie.update_layout(height=350, margin=dict(t=30, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("🔴", risk_counts.get('Critical', 0))
            m2.metric("🟡", risk_counts.get('Warning', 0))
            m3.metric("🟢", risk_counts.get('Normal', 0))
        
        with col2:
            st.markdown("### 🌡️ Profil Capteurs/Risque")
            data_train['risk'] = risk_levels
            heatmap_data = data_train.groupby('risk')[sensor_cols].mean()
            fig_heatmap = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, colorscale='RdYlGn_r'))
            fig_heatmap.update_layout(height=350, margin=dict(t=30, b=50))
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("---")
        
        # ROW 2: CORRELATION + ANOMALIES
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔗 Corrélation Top 10 Capteurs")
            corr_matrix = data_train[sensor_cols[:10]].corr()
            fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu', zmid=0))
            fig_corr.update_layout(height=400, margin=dict(t=30, b=50))
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.markdown("### 🔍 Anomalies Z-Score (S11)")
            s11_data = data_train['S11'].values
            z_scores = np.abs((s11_data - np.mean(s11_data)) / np.std(s11_data))
            anomalies = z_scores > 3
            anomaly_pct = (anomalies.sum() / len(anomalies)) * 100
            fig_z = px.histogram(z_scores, nbins=50, color_discrete_sequence=[COLOR_PALETTE['primary']])
            fig_z.add_vline(x=3, line_dash="dash", line_color=COLOR_PALETTE['danger'])
            fig_z.update_layout(height=400, xaxis_title="Z-Score", yaxis_title="Fréquence", margin=dict(t=30, b=50))
            st.plotly_chart(fig_z, use_container_width=True)
            a1, a2 = st.columns(2)
            a1.metric("Anomalies", int(anomalies.sum()))
            a2.metric("% Anomalies", f"{anomaly_pct:.2f}%")
        
        st.markdown("---")
        
        # ROW 3: ALERTS
        st.markdown("### 📋 Alertes Récentes (Top 20)")
        alerts_df = pd.DataFrame({'Moteur': data_train['unit_id'], 'Cycle': data_train['cycles'], 'Niveau': risk_levels})
        alerts_recent = alerts_df[alerts_df['Niveau'].isin(['Critical','Warning'])].sort_values('Cycle', ascending=False).head(20)
        st.dataframe(alerts_recent, use_container_width=True, height=250)
        
    except Exception as e:
        st.error(f"⚠️ Erreur: {e}")

elif page == "🎯 Dashboard Maintenance Prédictive":
    st.subheader("🎯 Dashboard Maintenance Prédictive — RUL & Planning Complet")
    
    try:
        data_train, _, _ = load_data()
        data_train = add_engine_targets(data_train)
        model, feat_cols, eval_metrics = train_rul_model(data_train)
        sensor_cols = [f'S{i+1}' for i in range(21)]
        
        # ROW 1: MODEL PERF
        st.markdown("### 📊 Performance Modèle")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{eval_metrics['MAE']:.1f} cycles")
        c2.metric("RMSE", f"{eval_metrics['RMSE']:.1f} cycles")
        c3.metric("Test Set", f"{eval_metrics['test_size']} obs")
        c4.metric("Modèle", "RandomForest")
        
        st.markdown("---")
        
        # ROW 2: RUL PRED + FEATURE ENG
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 Prédiction RUL")
            selected_motor = st.selectbox("Moteur:", sorted(data_train['unit_id'].unique())[:20], index=0)
            df_motor = data_train[data_train['unit_id'] == selected_motor].sort_values('cycles').copy()
            df_motor['rul_predicted'] = model.predict(df_motor[feat_cols].values)
            df_motor['rul_actual'] = df_motor['rul_true']
            residuals = df_motor['rul_predicted'] - df_motor['rul_actual']
            ci = 1.96 * np.std(residuals)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_motor['cycles'], y=df_motor['rul_predicted'] + ci, mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df_motor['cycles'], y=df_motor['rul_predicted'] - ci, fill='tonexty', mode='lines',
                                    fillcolor='rgba(52, 152, 219, 0.2)', line=dict(width=0), name='IC 95%'))
            fig.add_trace(go.Scatter(x=df_motor['cycles'], y=df_motor['rul_predicted'], mode='lines+markers', name='RUL Prédite',
                                    line=dict(color=COLOR_PALETTE['primary'], width=3)))
            fig.add_trace(go.Scatter(x=df_motor['cycles'], y=df_motor['rul_actual'], mode='lines', name='RUL Réelle',
                                    line=dict(color=COLOR_PALETTE['secondary'], dash='dash', width=2)))
            fig.add_hline(y=RUL_THRESHOLD_CRITICAL, line_dash="dot", line_color=COLOR_PALETTE['danger'], annotation_text="⚠️ Critique")
            fig.add_hline(y=RUL_THRESHOLD_WARNING, line_dash="dot", line_color=COLOR_PALETTE['warning'], annotation_text="⚠️ Alerte")
            fig.update_layout(height=400, xaxis_title="Cycles", yaxis_title="RUL", hovermode='x unified', margin=dict(t=30, b=50))
            st.plotly_chart(fig, use_container_width=True)
            
            rul_current = df_motor['rul_predicted'].iloc[-1]
            status = "🔴" if rul_current <= RUL_THRESHOLD_CRITICAL else "🟡" if rul_current <= RUL_THRESHOLD_WARNING else "🟢"
            s1, s2, s3 = st.columns(3)
            s1.metric("RUL Actuel", f"{rul_current:.1f}")
            s2.metric("Confiance ±", f"{ci:.1f}")
            s3.metric("Statut", status)
        
        with col2:
            st.markdown("### 🧩 Feature Engineering")
            sensor_choice = st.selectbox("Capteur:", sensor_cols, index=10)
            window = st.slider("Fenêtre:", 3, 60, 15)
            df_fe = df_motor.copy()
            df_fe[f'{sensor_choice}_roll'] = df_fe[sensor_choice].rolling(window=window, min_periods=1).mean()
            fig_fe = go.Figure()
            fig_fe.add_trace(go.Scatter(x=df_fe['cycles'], y=df_fe[sensor_choice], mode='lines', name=f"{sensor_choice} (brut)",
                                       line=dict(color=COLOR_PALETTE['primary'])))
            fig_fe.add_trace(go.Scatter(x=df_fe['cycles'], y=df_fe[f'{sensor_choice}_roll'], mode='lines', name=f"{sensor_choice} (lissé)",
                                       line=dict(color=COLOR_PALETTE['secondary'])))
            fig_fe.update_layout(height=400, xaxis_title="Cycles", yaxis_title="Valeur", margin=dict(t=30, b=50))
            st.plotly_chart(fig_fe, use_container_width=True)
        
        st.markdown("---")
        
        # ROW 3: MAINTENANCE TIMELINE
        st.markdown("### 📅 Timeline Maintenance (Top 15)")
        last_obs = data_train.sort_values('cycles').groupby('unit_id').tail(1).copy()
        last_obs['rul_pred'] = model.predict(last_obs[feat_cols].values)
        last_obs['priorite'] = np.where(last_obs['rul_pred'] <= RUL_THRESHOLD_CRITICAL, 'Critique',
                                 np.where(last_obs['rul_pred'] <= RUL_THRESHOLD_WARNING, 'Alerte', 'Normal'))
        plan = last_obs[['unit_id', 'cycles', 'rul_pred', 'priorite']].copy()
        plan['cycle_maintenance'] = (plan['cycles'] + np.maximum(plan['rul_pred'] - 5, 0)).round(0).astype(int)
        plan = plan.sort_values(['priorite', 'rul_pred'], ascending=[True, True]).head(15)
        plan.rename(columns={'unit_id': 'Moteur', 'cycles': 'Cycle actuel', 'rul_pred': 'RUL', 'priorite': 'Priorité'}, inplace=True)
        st.dataframe(plan[['Moteur', 'Cycle actuel', 'RUL', 'Priorité', 'cycle_maintenance']], use_container_width=True, height=350)
        
    except Exception as e:
        st.error(f"⚠️ Erreur: {e}")

elif page == "📈 Dashboard Analyse & Insights":
    st.subheader("📈 Dashboard Analyse & Insights — EDA, Clustering, Anomalies")
    
    try:
        data_train, _, _ = load_data()
        sensor_cols = [f'S{i+1}' for i in range(21)]
        
        # ROW 1: SENSOR STATS + CYCLES
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Variabilité Capteurs")
            sensor_stats = data_train[sensor_cols].describe().T
            fig_box = px.box(sensor_stats, y='std', color_discrete_sequence=[COLOR_PALETTE['primary']])
            fig_box.update_layout(height=350, yaxis_title="Écart-type", showlegend=False, margin=dict(t=30, b=50))
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Distribution Cycles")
            cycles_per_unit = data_train.groupby('unit_id')['cycles'].max()
            fig_hist = px.histogram(cycles_per_unit, nbins=30, color_discrete_sequence=[COLOR_PALETTE['primary']])
            fig_hist.update_layout(height=350, xaxis_title="Cycles", yaxis_title="Nb moteurs", showlegend=False, margin=dict(t=30, b=50))
            st.plotly_chart(fig_hist, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Moy", f"{cycles_per_unit.mean():.0f}")
            c2.metric("Min", f"{cycles_per_unit.min():.0f}")
            c3.metric("Max", f"{cycles_per_unit.max():.0f}")
        
        st.markdown("---")
        
        # ROW 2: CLUSTERING
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 🔗 Clustering (PCA)")
            k = st.slider("Clusters:", 2, 8, 3)
            agg_dict = {col: ['mean', 'std'] for col in sensor_cols}
            agg_df = data_train.groupby('unit_id').agg({**{'cycles': 'max'}, **agg_dict})
            agg_df.columns = ['_'.join([c for c in col if c]) for col in agg_df.columns.values]
            agg_df = agg_df.reset_index()
            feature_cols = [c for c in agg_df.columns if c not in ['unit_id']]
            scaler = StandardScaler()
            X = scaler.fit_transform(agg_df[feature_cols])
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X)
            explained = pca.explained_variance_ratio_.sum() * 100
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = km.fit_predict(X)
            cycles_max_series = agg_df['cycles_max'] if 'cycles_max' in agg_df.columns else data_train.groupby('unit_id')['cycles'].max().values
            df_vis = pd.DataFrame({'unit_id': agg_df['unit_id'], 'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 
                                  'cluster': clusters, 'cycles_max': cycles_max_series})
            fig_scatter = px.scatter(df_vis, x='PC1', y='PC2', color=df_vis['cluster'].astype(str),
                                    hover_data=['unit_id', 'cycles_max'], color_discrete_sequence=px.colors.qualitative.Set2)
            fig_scatter.update_layout(height=400, margin=dict(t=30, b=50))
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(f"Variance: {explained:.1f}%")
        
        with col2:
            st.markdown("### 📐 Qualité")
            try:
                sil = silhouette_score(X, clusters)
                dbi = davies_bouldin_score(X, clusters)
                m1, m2 = st.columns(2)
                m1.metric("Silhouette", f"{sil:.3f}")
                m2.metric("Davies-Bouldin", f"{dbi:.3f}")
            except:
                st.caption("Non calculé")
            
            st.markdown("#### Répartition")
            counts = df_vis['cluster'].value_counts().sort_index()
            fig_bar = px.bar(x=counts.index.astype(str), y=counts.values, color=counts.index.astype(str),
                            color_discrete_sequence=px.colors.qualitative.Set2)
            fig_bar.update_layout(height=250, showlegend=False, xaxis_title="Cluster", yaxis_title="Nb", margin=dict(t=20, b=50))
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        
        # ROW 3: ANOMALIES + SETTINGS
        st.markdown("### 🔍 Détection Anomalies (Isolation Forest)")
        col1, col2 = st.columns([1, 2])
        with col1:
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            anomalies_if = iso_forest.fit_predict(data_train[sensor_cols])
            n_anomalies = (anomalies_if == -1).sum()
            anomaly_pct = (n_anomalies / len(anomalies_if)) * 100
            a1, a2, a3 = st.columns(3)
            a1.metric("Anomalies", int(n_anomalies))
            a2.metric("% Anom", f"{anomaly_pct:.2f}%")
            a3.metric("Normaux", int((anomalies_if == 1).sum()))
        
        with col2:
            st.markdown("#### Paramètres Opérationnels")
            settings_cols = ['setting1', 'setting2', 'setting3']
            fig_settings = make_subplots(rows=1, cols=3, subplot_titles=settings_cols)
            for i, col in enumerate(settings_cols, 1):
                fig_settings.add_trace(go.Histogram(x=data_train[col], nbinsx=30, marker_color=COLOR_PALETTE['primary']), row=1, col=i)
            fig_settings.update_layout(height=250, showlegend=False, margin=dict(t=50, b=50))
            st.plotly_chart(fig_settings, use_container_width=True)
        
    except Exception as e:
        st.error(f"⚠️ Erreur: {e}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #95a5a6; font-size: 12px;'>
    <p>🛩️ Tableau de Bord AeroMaintain | NASA C-MAPSS FD001 | Maintenance Prédictive</p>
    <p>Streamlit • Plotly • Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
