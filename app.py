"""
🛩️ AeroMaintain Dashboard - Streamlit Application
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
    .main {
        padding-top: 0rem;
    }
    h1 {
        color: #3498db;
        text-align: center;
        margin-bottom: 1rem;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
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
    """Load and preprocess NASA C-MAPSS data"""
    dataset_path = Path('dataset')
    
    # Load data
    train_data = pd.read_csv(
        dataset_path / 'train_FD001.txt',
        sep=r'\s+',
        header=None,
        names=['unit_id', 'cycles', 'setting1', 'setting2', 'setting3'] + 
              [f'S{i+1}' for i in range(21)]
    )
    
    test_data = pd.read_csv(
        dataset_path / 'test_FD001.txt',
        sep=r'\s+',
        header=None,
        names=['unit_id', 'cycles', 'setting1', 'setting2', 'setting3'] + 
              [f'S{i+1}' for i in range(21)]
    )
    
    rul_data = pd.read_csv(
        dataset_path / 'RUL_FD001.txt',
        sep=r'\s+',
        header=None,
        names=['RUL']
    )
    
    return train_data, test_data, rul_data

# ==================== HELPERS & CACHING ====================
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
    """Train a simple RandomForest RUL regressor on engine rows.
    Split by engines to reduce leakage. Returns (model, feature_cols, eval_metrics).
    """
    sensor_cols = [f'S{i+1}' for i in range(21)]
    setting_cols = ['setting1', 'setting2', 'setting3']
    feat_cols = setting_cols + sensor_cols

    # Engine-level split
    engines = df['unit_id'].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(engines)
    cut = int(0.8 * len(engines))
    train_engines = set(engines[:cut])
    test_engines = set(engines[cut:])

    train_df = df[df['unit_id'].isin(train_engines)]
    test_df = df[df['unit_id'].isin(test_engines)]

    X_train = train_df[feat_cols].values
    y_train = train_df['rul_true'].values
    X_test = test_df[feat_cols].values
    y_test = test_df['rul_true'].values

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics = {"MAE": mae, "RMSE": rmse, "test_size": int(len(y_test))}
    return model, feat_cols, metrics

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

if page == "📊 Dashboard Executive":
    st.subheader("� Dashboard Executive — Vue d'Ensemble Stratégique")

    try:
        data_train, _, _ = load_data()
        data_train = add_engine_targets(data_train)
        sensor_cols = [f'S{i+1}' for i in range(21)]

        # Calculate all metrics
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
        
        # === ROW 1: TOP KPIs ===
        st.markdown("### 💼 Indicateurs Clés de Performance")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🛩️ Flotte Totale", FLEET_SIZE)
        col2.metric("⚙️ Analysés", int(nb_moteurs))
        col3.metric("⏱️ Cycles Moyens", f"{mean_cycles:.0f}")
        col4.metric("💰 Coûts Évités", f"€{savings:,}".replace(","," "))
        col5.metric("📈 ROI", f"{roi_pct:.0f}%")
        
        st.markdown("---")
        
        # === ROW 2: RISK PIE + LIFECYCLE EVOLUTION ===
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎯 État de la Flotte")
            risk_counts = pd.Series(levels).value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                hole=0.4,
                color_discrete_map={'Critique': COLOR_PALETTE['danger'], 
                                    'Alerte': COLOR_PALETTE['warning'], 
                                    'Normal': COLOR_PALETTE['secondary']}
            )
            fig_pie.update_layout(height=350, margin=dict(t=30, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("🔴", int(risk_counts.get('Critique', 0)))
            m2.metric("🟡", int(risk_counts.get('Alerte', 0)))
            m3.metric("🟢", int(risk_counts.get('Normal', 0)))
        
        with col2:
            st.markdown("### 📊 Évolution Risques (Cycle de Vie)")
            df_levels = data_train[['unit_id', 'cycles', 'cycles_max', 'progress_rel']].copy()
            df_levels['risk'] = levels
            df_levels['bin'] = pd.cut(df_levels['progress_rel'], bins=np.linspace(0, 1, 21), 
                                     labels=[f"{int(b*5)}%" for b in range(20)])
            evo = df_levels.groupby(['bin', 'risk']).size().reset_index(name='count')
            evo_pivot = evo.pivot(index='bin', columns='risk', values='count').fillna(0).reset_index()
            evo_pivot = evo_pivot.sort_values('bin')
            
            fig_area = px.area(
                evo_pivot,
                x='bin', y=['Normal', 'Alerte', 'Critique'],
                color_discrete_map={'Critique': COLOR_PALETTE['danger'], 
                                   'Alerte': COLOR_PALETTE['warning'], 
                                   'Normal': COLOR_PALETTE['secondary']}
            )
            fig_area.update_layout(height=350, xaxis_title="Progression", 
                                  yaxis_title="Moteurs", showlegend=True,
                                  margin=dict(t=30, b=50))
            st.plotly_chart(fig_area, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Évolution des niveaux de risque (cycle relatif)")
        fig_area = px.area(
            evo_pivot,
            x='bin', y=['Critique', 'Alerte', 'Normal'],
            title="Évolution du nombre de moteurs par niveau de risque",
            color_discrete_map={'Critique': COLOR_PALETTE['danger'], 'Alerte': COLOR_PALETTE['warning'], 'Normal': COLOR_PALETTE['secondary']}
        )
        st.plotly_chart(fig_area, use_container_width=True)

        st.markdown("---")
        st.markdown("### Alerte moteurs critiques (instantané ~70% de vie)")
        snap = data_train[(data_train['progress_rel'] >= 0.7) & (data_train['progress_rel'] < 0.75)].copy()
        if len(snap) > 0:
            # risk on snapshot
            snap_scaled = scaler.fit_transform(snap[sensor_cols])
            snap_std = np.std(snap_scaled, axis=1)
            snap_levels = np.where(snap_std > np.percentile(snap_std, 90), 'Critique', np.where(snap_std > np.percentile(snap_std, 70), 'Alerte', 'Normal'))
            snap['risk'] = snap_levels
            crit_engines = sorted(snap.loc[snap['risk'] == 'Critique', 'unit_id'].unique().tolist())
            if crit_engines:
                st.warning(f"Moteurs à risque critique (snapshot): {', '.join(map(str, crit_engines[:15]))}{' …' if len(crit_engines)>15 else ''}")
            else:
                st.success("Aucun moteur critique dans cette fenêtre.")
        else:
            st.info("Fenêtre snapshot vide pour certains jeux – affichage omis.")

        st.markdown("---")
        st.markdown("### KPIs financiers (modèle simple)")
        # Simple cost model assumptions
        cost_unplanned = 120_000
        cost_planned = 45_000
        # Engines that would cross critical threshold soon (<=30 cycles) at last row
        nearing = last_rows[last_rows['rul_true'] <= 30]
        avoided_per_engine = max(cost_unplanned - cost_planned, 0)
        savings = int(len(nearing) * avoided_per_engine)
        roi_pct = (avoided_per_engine / cost_planned) * 100 if cost_planned > 0 else 0
        f1, f2, f3 = st.columns(3)
        f1.metric("Coût évité (estimé)", f"€ {savings:,}".replace(",", " "))
        f2.metric("ROI maintenance préventive", f"{roi_pct:.0f}%")
        f3.metric("Moteurs < 30 cycles RUL", int(len(nearing)))

        st.caption("Hypothèses: coût panne non planifiée=120k€, maintenance planifiée=45k€.")

    except Exception as e:
        st.error(f"⚠️ Erreur dans la Vue d'Ensemble Executive : {e}")

if page == "🏠 Accueil":
    st.sidebar.markdown("""
**Dataset NASA C-MAPSS FD001**
- ~100 moteurs turbofan (simulation)
- 21 capteurs par moteur
- Données de capteurs jusqu'à la panne

### 🎯 Objectifs
- Prédire la RUL (Durée de Vie Restante)
- Détecter les anomalies capteurs
- Optimiser la maintenance (préventive vs corrective)
""")

# ==================== MAIN HEADER ====================
st.markdown("# 🛩️ Tableau de Bord AeroMaintain - Prédiction Intelligente de Maintenance")
st.markdown("**Maintenance Prédictive pour Moteurs Turbofan utilisant le Machine Learning**")
st.markdown("---")

# ==================== PAGE ROUTING ====================

if page == "🏠 Accueil":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "📦 Moteurs Totaux",
            FLEET_SIZE,
            "NASA C-MAPSS FD001"
        )
    with col2:
        st.metric(
            "📊 Capteurs",
            21,
            "par moteur"
        )
    with col3:
        st.metric(
            "📈 Points de Données",
            "~20 000",
            "cycles totaux"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✨ Fonctionnalités Clés")
        st.write("""
        ✅ **Prédiction RUL** - Estimer la durée de vie utile restante
        
        ✅ **Détection d'Anomalies** - Identifier les anomalies de capteurs
        
        ✅ **Classification des Risques** - Catégoriser par urgence
        
        ✅ **Analyse de Clustering** - Segmenter par état de santé
        
        ✅ **Surveillance en Temps Réel** - Suivre la dégradation
        """)
    
    with col2:
        st.subheader("🎯 Avantages")
        st.write("""
        💰 **Économies** - Prévenir les pannes inattendues
        
        ⏰ **Optimisation** - Planifier la maintenance proactivement
        
        🚀 **Fiabilité** - Maximiser le temps de fonctionnement
        
        📊 **Décisions Basées sur les Données** - Insights ML
        
        🛡️ **Mitigation des Risques** - Système d'alerte précoce
        """)
    
    st.markdown("---")
    st.markdown("### 📚 Informations sur les Données")
    
    try:
        data_train, data_test, rul = load_data()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Échantillons d'Entraînement", len(data_train))
        with col2:
            st.metric("Échantillons de Test", len(data_test))
        with col3:
            st.metric("Moteurs Uniques", data_train['unit_id'].nunique())
        with col4:
            st.metric("Max Cycles/Moteur", int(data_train.groupby('unit_id')['cycles'].max().max()))
            
    except Exception as e:
        st.error(f"⚠️ Erreur lors du chargement des données : {e}")

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
                from sklearn.metrics import silhouette_score, davies_bouldin_score
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

elif page == "🧩 Ingénierie des Variables":
    st.subheader("🧩 Ingénierie des Variables (Feature Engineering)")

    try:
        data_train, _, _ = load_data()
        sensor_cols = [f'S{i+1}' for i in range(21)]

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            engine_id = st.selectbox("ID du Moteur :", sorted(data_train['unit_id'].unique())[:50])
        with col2:
            sensor = st.selectbox("Capteur :", sensor_cols, index=10)
        with col3:
            window = st.slider("Fenêtre de lissage (rolling)", 3, 60, 15, step=1)

        df_e = data_train[data_train['unit_id'] == engine_id].sort_values('cycles').copy()
        df_e[f'{sensor}_roll_mean'] = df_e[sensor].rolling(window=window, min_periods=1).mean()
        df_e[f'{sensor}_roll_std'] = df_e[sensor].rolling(window=window, min_periods=1).std().fillna(0)
        df_e[f'{sensor}_diff'] = df_e[sensor].diff().fillna(0)

        fig_fe = go.Figure()
        fig_fe.add_trace(go.Scatter(x=df_e['cycles'], y=df_e[sensor], mode='lines', name=f"{sensor} (brut)",
                                     line=dict(color=COLOR_PALETTE['primary'])))
        fig_fe.add_trace(go.Scatter(x=df_e['cycles'], y=df_e[f'{sensor}_roll_mean'], mode='lines',
                                     name=f"{sensor} (moyenne glissante)",
                                     line=dict(color=COLOR_PALETTE['secondary'])))
        fig_fe.update_layout(title=f"{sensor} - Moteur #{engine_id}", xaxis_title="Cycles", yaxis_title="Valeur")
        st.plotly_chart(fig_fe, use_container_width=True)

        st.write("### Aperçu des nouvelles variables")
        st.dataframe(df_e[["cycles", sensor, f"{sensor}_roll_mean", f"{sensor}_roll_std", f"{sensor}_diff"]].head(20),
                     use_container_width=True)

        st.info("Astuce: ces variables enrichies peuvent alimenter clustering, classification et modèles RUL.")

    except Exception as e:
        st.error(f"⚠️ Erreur dans l'Ingénierie des Variables : {e}")

elif page == "🔗 Clustering & Segmentation":
    st.subheader("🔗 Clustering & Segmentation des Moteurs")

    try:
        data_train, _, _ = load_data()
        sensor_cols = [f'S{i+1}' for i in range(21)]

        # Agrégation au niveau moteur (moyenne et écart-type des capteurs, cycles max)
        agg_dict = {col: ['mean', 'std'] for col in sensor_cols}
        agg_df = data_train.groupby('unit_id').agg({**{'cycles': 'max'}, **agg_dict})
        agg_df.columns = ['_'.join([c for c in col if c]) for col in agg_df.columns.values]
        agg_df = agg_df.reset_index().rename(columns={'cycles_max': 'cycles_max'})

        feature_cols = [c for c in agg_df.columns if c not in ['unit_id']]
        scaler = StandardScaler()
        X = scaler.fit_transform(agg_df[feature_cols])

        # PCA pour visualisation
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_.sum() * 100

        k = st.slider("Nombre de clusters (k)", 2, 8, 3)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(X)

        cycles_max_series = agg_df['cycles_max'] if 'cycles_max' in agg_df.columns else data_train.groupby('unit_id')['cycles'].max().values
        df_vis = pd.DataFrame({
            'unit_id': agg_df['unit_id'],
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'cluster': clusters,
            'cycles_max': cycles_max_series
        })

        col1, col2 = st.columns([2, 1])
        with col1:
            fig_scatter = px.scatter(
                df_vis,
                x='PC1', y='PC2', color=df_vis['cluster'].astype(str),
                hover_data=['unit_id', 'cycles_max'],
                title=f"Projection PCA (2D) — Variance expliquée ≈ {explained:.1f}%",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={'color': 'cluster'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            counts = df_vis['cluster'].value_counts().sort_index()
            st.metric("Nombre de clusters", k)
            st.metric("Moteurs segmentés", int(len(df_vis)))
            fig_bar = px.bar(x=counts.index.astype(str), y=counts.values, labels={'x': 'Cluster', 'y': 'Nb moteurs'},
                             title="Répartition par cluster",
                             color=counts.index.astype(str), color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        # Quality metrics (on original feature space X)
        try:
            sil = silhouette_score(X, clusters)
            dbi = davies_bouldin_score(X, clusters)
            m1, m2 = st.columns(2)
            m1.metric("Silhouette Score", f"{sil:.3f}")
            m2.metric("Davies-Bouldin", f"{dbi:.3f}")
        except Exception:
            st.caption("Scores de clustering non calculés (échantillon trop petit ou configuration invalide).")

        st.write("### Aperçu des caractéristiques agrégées (niveau moteur)")
        st.dataframe(agg_df.head(10), use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Erreur dans le Clustering : {e}")

elif page == "📈 Analyse Exploratoire":
    st.subheader("📊 Analyse Exploratoire des Données")
    
    try:
        data_train, _, _ = load_data()
        
        analysis_type = st.selectbox(
            "Sélectionnez le Type d'Analyse:",
            ["Statistiques des Capteurs", "Analyse de Corrélation", "Paramètres Opérationnels", "Distribution RUL"]
        )
        
        if analysis_type == "Statistiques des Capteurs":
            st.write("### Résumé des Données des Capteurs")
            sensor_cols = [f'S{i+1}' for i in range(21)]
            
            st.dataframe(
                data_train[sensor_cols].describe().T,
                use_container_width=True
            )
            
            # Visualize sensor variance
            fig = px.box(
                data_train[sensor_cols].describe().T,
                y='std',
                title='Variabilité des Capteurs (Écart-type)',
                labels={'std': 'Écart-type'},
                color_discrete_sequence=[COLOR_PALETTE['primary']]
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Analyse de Corrélation":
            st.write("### Matrice de Corrélation des Capteurs")
            sensor_cols = [f'S{i+1}' for i in range(21)]
            
            corr_matrix = data_train[sensor_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Corrélation")
            ))
            
            fig.update_layout(
                title="Carte Thermique de Corrélation des Capteurs",
                width=800,
                height=800
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Paramètres Opérationnels":
            st.write("### Distribution des Paramètres Opérationnels")
            
            settings_cols = ['setting1', 'setting2', 'setting3']
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=settings_cols
            )
            
            for i, col in enumerate(settings_cols, 1):
                fig.add_trace(
                    go.Histogram(
                        x=data_train[col],
                        name=col,
                        nbinsx=30,
                        marker_color=COLOR_PALETTE['primary']
                    ),
                    row=1, col=i
                )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Distribution RUL":
            st.write("### Cycles de Fonctionnement du Moteur")
            
            cycles_per_unit = data_train.groupby('unit_id')['cycles'].max()
            
            fig = px.histogram(
                cycles_per_unit,
                nbins=30,
                title="Distribution des Cycles de Fonctionnement par Moteur",
                labels={'value': 'Cycles', 'count': 'Nombre de Moteurs'},
                color_discrete_sequence=[COLOR_PALETTE['primary']]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cycles Moyens", f"{cycles_per_unit.mean():.0f}")
            with col2:
                st.metric("Cycles Min", f"{cycles_per_unit.min():.0f}")
            with col3:
                st.metric("Cycles Max", f"{cycles_per_unit.max():.0f}")
                
    except Exception as e:
        st.error(f"⚠️ Erreur dans l'EDA : {e}")

elif page == "🎯 Prédictions":
    st.subheader("🎯 Tableau de Bord de Prédiction RUL")
    
    try:
        data_train, _, _ = load_data()
        data_train = add_engine_targets(data_train)
        model, feat_cols, eval_metrics = train_rul_model(data_train)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("### Prédiction RUL Interactive")
            st.info("📌 Sélectionnez un moteur pour voir sa prédiction RUL au fil du temps")
            
        with col2:
            selected_motor = st.selectbox(
                "ID du Moteur:",
                sorted(data_train['unit_id'].unique())[:20],
                index=0
            )
        
        df_motor = data_train[data_train['unit_id'] == selected_motor].sort_values('cycles').copy()
        
        # Predict RUL with trained model
        df_motor['rul_predicted'] = model.predict(df_motor[feat_cols].values)
        df_motor['rul_actual'] = df_motor['rul_true']
        
        # Calculate confidence interval (residual dispersion)
        residuals = df_motor['rul_predicted'] - df_motor['rul_actual']
        std_error = np.std(residuals)
        ci = 1.96 * std_error
        
        # Create figure
        fig = go.Figure()
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=df_motor['cycles'],
            y=df_motor['rul_predicted'] + ci,
            mode='lines',
            name='IC Upper (95%)',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df_motor['cycles'],
            y=df_motor['rul_predicted'] - ci,
            fill='tonexty',
            mode='lines',
            name='Confidence Interval 95%',
            fillcolor='rgba(52, 152, 219, 0.2)',
            line=dict(width=0)
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=df_motor['cycles'],
            y=df_motor['rul_predicted'],
            mode='lines+markers',
            name='RUL Predicted',
            line=dict(color=COLOR_PALETTE['primary'], width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_motor['cycles'],
            y=df_motor['rul_actual'],
            mode='lines',
            name='RUL Actual',
            line=dict(color=COLOR_PALETTE['secondary'], dash='dash', width=2)
        ))
        
        # Add thresholds
        fig.add_hline(
            y=RUL_THRESHOLD_CRITICAL,
            line_dash="dot",
            line_color=COLOR_PALETTE['danger'],
            annotation_text="⚠️ Critique"
        )
        
        fig.add_hline(
            y=RUL_THRESHOLD_WARNING,
            line_dash="dot",
            line_color=COLOR_PALETTE['warning'],
            annotation_text="⚠️ Alerte"
        )
        
        fig.update_layout(
            title=f"Prédiction RUL - Moteur #{selected_motor}",
            xaxis_title="Cycles de Fonctionnement",
            yaxis_title="RUL (Cycles Restants)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RUL Actuel", f"{df_motor['rul_predicted'].iloc[-1]:.1f} cycles")
        with col2:
            st.metric("Confiance ±", f"{ci:.1f} cycles")
        with col3:
            status = "🔴 CRITIQUE" if df_motor['rul_predicted'].iloc[-1] <= RUL_THRESHOLD_CRITICAL else "🟡 ALERTE" if df_motor['rul_predicted'].iloc[-1] <= RUL_THRESHOLD_WARNING else "🟢 NORMAL"
            st.metric("Statut", status)
        with col4:
            st.metric("Cycles Totaux", f"{df_motor['cycles'].max():.0f}")
        
        st.caption(f"Modèle RF — MAE: {eval_metrics['MAE']:.1f} | RMSE: {eval_metrics['RMSE']:.1f} (test engines: {eval_metrics['test_size']})")
        
        st.markdown("---")
        st.markdown("### Timeline des maintenances recommandées (Top moteurs)")
        # Current RUL prediction at last observation per engine
        last_obs = data_train.sort_values('cycles').groupby('unit_id').tail(1).copy()
        last_obs['rul_pred'] = model.predict(last_obs[feat_cols].values)
        last_obs['priorite'] = np.where(last_obs['rul_pred'] <= RUL_THRESHOLD_CRITICAL, 'Critique',
                                 np.where(last_obs['rul_pred'] <= RUL_THRESHOLD_WARNING, 'Alerte', 'Normal'))
        plan = last_obs[['unit_id', 'cycles', 'rul_pred', 'priorite', 'cycles_max']].copy()
        plan['cycle_maintenance'] = (plan['cycles'] + np.maximum(plan['rul_pred'] - 5, 0)).round(0).astype(int)
        plan = plan.sort_values(['priorite', 'rul_pred'], ascending=[True, True]).head(15)
        plan.rename(columns={'unit_id': 'Moteur', 'cycles': 'Cycle actuel', 'rul_pred': 'RUL prédite', 'priorite': 'Priorité'}, inplace=True)
        st.dataframe(plan[['Moteur', 'Cycle actuel', 'RUL prédite', 'Priorité', 'cycle_maintenance']], use_container_width=True)
        st.caption("Heuristique: planifier la maintenance avant la fenêtre critique (marge de 5 cycles).")
            
    except Exception as e:
        st.error(f"⚠️ Erreur dans les Prédictions : {e}")

elif page == "🧠 Classification du Risque":
    st.subheader("🧠 Classification du Risque (Règles Simples)")

    try:
        data_train, _, _ = load_data()
        sensor_cols = [f'S{i+1}' for i in range(21)]

        # Score de risque simple: écart-type agrégé par ligne, puis moyenne par moteur
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_train[sensor_cols])
        row_std = np.std(data_scaled, axis=1)
        risk_by_engine = pd.DataFrame({
            'unit_id': data_train['unit_id'],
            'row_risk': row_std
        }).groupby('unit_id')['row_risk'].mean().reset_index(name='risk_score')

        q70 = risk_by_engine['risk_score'].quantile(0.70)
        q90 = risk_by_engine['risk_score'].quantile(0.90)

        def map_level(x):
            if x > q90:
                return 'Critique'
            if x > q70:
                return 'Alerte'
            return 'Normal'

        risk_by_engine['niveau_risque'] = risk_by_engine['risk_score'].apply(map_level)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig_risk = px.histogram(risk_by_engine, x='risk_score', color='niveau_risque',
                                     nbins=30,
                                     title="Distribution des scores de risque (niveau moteur)",
                                     color_discrete_map={'Critique': COLOR_PALETTE['danger'],
                                                         'Alerte': COLOR_PALETTE['warning'],
                                                         'Normal': COLOR_PALETTE['secondary']})
            st.plotly_chart(fig_risk, use_container_width=True)

        with col2:
            counts = risk_by_engine['niveau_risque'].value_counts()
            st.metric("🔴 Critiques", int(counts.get('Critique', 0)))
            st.metric("🟡 Alertes", int(counts.get('Alerte', 0)))
            st.metric("🟢 Normaux", int(counts.get('Normal', 0)))

        st.markdown("---")
        st.write("### Tableau des niveaux de risque par moteur")
        st.dataframe(risk_by_engine.sort_values('risk_score', ascending=False).head(20), use_container_width=True)

        st.info("Ce classificateur par règles sert de base. On peut le remplacer par un modèle supervisé lorsqu'une vérité terrain (labels) est disponible.")

    except Exception as e:
        st.error(f"⚠️ Erreur dans la Classification du Risque : {e}")

elif page == "📈 KPIs & Évaluation":
    st.subheader("📈 KPIs & Évaluation")

    try:
        data_train, _, _ = load_data()
        sensor_cols = [f'S{i+1}' for i in range(21)]

        # KPIs globaux
        nb_moteurs = data_train['unit_id'].nunique()
        cycles_par_moteur = data_train.groupby('unit_id')['cycles'].max()
        moyenne_cycles = cycles_par_moteur.mean()
        mediane_cycles = cycles_par_moteur.median()

        # Anomalies (méthode Z-score sur S11)
        s11 = data_train['S11'].values
        z = np.abs((s11 - np.mean(s11)) / (np.std(s11) + 1e-8))
        pct_anomalies = (z > 3).mean() * 100

        # PCA couverture sur agrégats
        agg = data_train.groupby('unit_id')[sensor_cols].mean()
        scaler = StandardScaler()
        Xagg = scaler.fit_transform(agg)
        pca = PCA(n_components=2, random_state=42)
        pca.fit(Xagg)
        pca_cov = pca.explained_variance_ratio_.sum() * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Moteurs", int(nb_moteurs))
        col2.metric("Cycles moyens (max)", f"{moyenne_cycles:.0f}")
        col3.metric("Cycles médians (max)", f"{mediane_cycles:.0f}")
        col4.metric("% anomalies S11 (Z>3)", f"{pct_anomalies:.2f}%")

        st.markdown("---")
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Couverture PCA (2D)", f"{pca_cov:.1f}%")
            fig_cycles = px.histogram(cycles_par_moteur, nbins=25,
                                      title="Distribution des cycles max par moteur",
                                      labels={'value': 'Cycles', 'count': 'Nb moteurs'},
                                      color_discrete_sequence=[COLOR_PALETTE['primary']])
            st.plotly_chart(fig_cycles, use_container_width=True)
        with col6:
            # Courbe ROC/PR non disponible sans labels; on affiche un radar KPI fictif (normalisé)
            kpi_df = pd.DataFrame({
                'KPI': ['Couverture PCA', 'Anomalies (faibles)', 'Dispersion cycles'],
                'val': [pca_cov/100, 1-(pct_anomalies/100), np.std(cycles_par_moteur)/ (np.mean(cycles_par_moteur)+1e-8)]
            })
            fig_bar = px.bar(kpi_df, x='KPI', y='val', title="KPI (normalisés)",
                             color='KPI', color_discrete_sequence=px.colors.qualitative.Set2,
                             range_y=[0, 1])
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.write("### Performance Modèle RUL (RandomForest)")
        df_targets = add_engine_targets(data_train)
        _, _, metrics = train_rul_model(df_targets)
        m1, m2 = st.columns(2)
        m1.metric("MAE", f"{metrics['MAE']:.1f}")
        m2.metric("RMSE", f"{metrics['RMSE']:.1f}")

    except Exception as e:
        st.error(f"⚠️ Erreur dans les KPIs : {e}")

elif page == "🔍 Détection d'Anomalies":
    st.subheader("🔍 Analyse de Détection d'Anomalies")
    
    try:
        data_train, _, _ = load_data()
        
        st.write("### Détection des Anomalies dans les Données des Capteurs")
        
        method = st.selectbox(
            "Méthode de Détection:",
            ["Z-Score", "Isolation Forest", "Corrélation Roulante"]
        )
        
        sensor_cols = [f'S{i+1}' for i in range(21)]
        
        if method == "Z-Score":
            st.write("""
            **Méthode Z-Score**: Détecte les valeurs aberrantes par écarts-types
            - Seuil : Les valeurs avec |Z| > 3 sont des anomalies
            - Représente seulement 0,3% de la distribution normale
            """)
            
            # Calculate Z-scores for S11
            s11_data = data_train['S11'].values
            mean_s11 = np.mean(s11_data)
            std_s11 = np.std(s11_data)
            z_scores = np.abs((s11_data - mean_s11) / std_s11)
            
            anomalies = z_scores > 3
            anomaly_pct = (anomalies.sum() / len(anomalies)) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anomalies Détectées", int(anomalies.sum()))
            with col2:
                st.metric("% Anomalies", f"{anomaly_pct:.2f}%")
            with col3:
                st.metric("Points Normaux", int((~anomalies).sum()))
            
            # Visualization
            fig = px.histogram(
                z_scores,
                nbins=50,
                title="Distribution Z-Score (Capteur S11)",
                labels={'value': 'Z-Score', 'count': 'Fréquence'},
                color_discrete_sequence=[COLOR_PALETTE['primary']]
            )
            
            fig.add_vline(x=3, line_dash="dash", line_color=COLOR_PALETTE['danger'], annotation_text="Seuil (Z=3)")
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif method == "Isolation Forest":
            st.write("""
            **Isolation Forest**: Détection d'anomalies par apprentissage automatique
            - Isole les anomalies par partitionnement aléatoire
            - Efficace pour les données haute-dimensionnelles
            """)
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            anomalies_if = iso_forest.fit_predict(data_train[sensor_cols])
            
            n_anomalies = (anomalies_if == -1).sum()
            anomaly_pct = (n_anomalies / len(anomalies_if)) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anomalies Détectées", int(n_anomalies))
            with col2:
                st.metric("% Anomalies", f"{anomaly_pct:.2f}%")
            with col3:
                st.metric("Points Normaux", int((anomalies_if == 1).sum()))
            
        else:  # Rolling Correlation
            st.write("""
            **Corrélation Roulante**: Surveille la dégradation de la relation des capteurs
            - Corrèle S11 vs S12 dans des fenêtres roulantes
            - Détecte quand la corrélation tombe en dessous du seuil
            """)
            
            s11_norm = (data_train['S11'] - data_train['S11'].min()) / (data_train['S11'].max() - data_train['S11'].min())
            s12_norm = (data_train['S12'] - data_train['S12'].min()) / (data_train['S12'].max() - data_train['S12'].min())
            
            rolling_corr = s11_norm.rolling(window=30).corr(s12_norm)
            
            fig = px.line(
                y=rolling_corr,
                title="Corrélation Roulante S11 vs S12 (Fenêtre=30)",
                labels={'y': 'Corrélation', 'x': 'Échantillon'},
                color_discrete_sequence=[COLOR_PALETTE['primary']]
            )
            
            fig.add_hline(y=-0.75, line_dash="dash", line_color=COLOR_PALETTE['warning'], annotation_text="Seuil d'Alerte")
            fig.add_hline(y=-0.60, line_dash="dash", line_color=COLOR_PALETTE['danger'], annotation_text="Seuil Critique")
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"⚠️ Erreur dans la Détection d'Anomalies : {e}")

elif page == "📊 Surveillance":
    st.subheader("📊 Tableau de Bord de Surveillance en Temps Réel")
    
    try:
        data_train, _, _ = load_data()
        
        st.write("### Aperçu de la Santé de la Flotte")
        
        # Calculate mock risk scores
        sensor_cols = [f'S{i+1}' for i in range(21)]
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_train[sensor_cols])
        
        # Simple risk scoring
        risk_scores = np.std(data_scaled, axis=1)
        
        # Assign risk levels
        risk_levels = []
        for score in risk_scores:
            if score > np.percentile(risk_scores, 90):
                risk_levels.append('Critical')
            elif score > np.percentile(risk_scores, 70):
                risk_levels.append('Warning')
            else:
                risk_levels.append('Normal')
        
        risk_counts = pd.Series(risk_levels).value_counts()
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Distribution des Risques de la Flotte",
                color_discrete_map={
                    'Critical': COLOR_PALETTE['danger'],
                    'Warning': COLOR_PALETTE['warning'],
                    'Normal': COLOR_PALETTE['secondary']
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Summary metrics
            st.metric("🔴 Moteurs Critiques", risk_counts.get('Critical', 0))
            st.metric("🟡 Moteurs en Alerte", risk_counts.get('Warning', 0))
            st.metric("🟢 Moteurs Normaux", risk_counts.get('Normal', 0))
        
        st.markdown("---")
        
        # Heatmap of sensor averages by risk level
        st.write("### Profils des Capteurs par Niveau de Risque")
        
        data_train['risk'] = risk_levels
        
        heatmap_data = data_train.groupby('risk')[sensor_cols].mean()
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn_r'
        ))
        
        fig_heatmap.update_layout(
            title="Valeurs Moyennes des Capteurs par Niveau de Risque",
            height=400
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("---")
        st.write("### Historique des alertes récentes")
        alerts_df = pd.DataFrame({
            'unit_id': data_train['unit_id'],
            'cycles': data_train['cycles'],
            'niveau': risk_levels
        })
        alerts_recent = alerts_df[alerts_df['niveau'].isin(['Critical','Warning'])].sort_values('cycles', ascending=False).head(20)
        alerts_recent.rename(columns={'unit_id':'Moteur','cycles':'Cycle','niveau':'Niveau'}, inplace=True)
        st.dataframe(alerts_recent, use_container_width=True)
        
    except Exception as e:
        st.error(f"⚠️ Erreur dans la Surveillance : {e}")

elif page == "🧾 Synthèse Business":
    st.subheader("🧾 Synthèse Business")

    try:
        data_train, _, _ = load_data()
        sensor_cols = [f'S{i+1}' for i in range(21)]

        nb_moteurs = data_train['unit_id'].nunique()
        cycles_par_moteur = data_train.groupby('unit_id')['cycles'].max()
        moyenne_cycles = cycles_par_moteur.mean()
        mediane_cycles = cycles_par_moteur.median()

        # Indicateurs risques simples (réutilisation logique du score std)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_train[sensor_cols])
        risk_scores = np.std(data_scaled, axis=1)
        seuil_crit = np.percentile(risk_scores, 90)
        seuil_warn = np.percentile(risk_scores, 70)
        niveaux = np.where(risk_scores > seuil_crit, 'Critique', np.where(risk_scores > seuil_warn, 'Alerte', 'Normal'))
        counts = pd.Series(niveaux).value_counts()

        st.markdown("### Points Clés")
        st.write("""
        - Flotte analysée: {nb} moteurs (jeu FD001 NASA C-MAPSS)
        - Charge opérationnelle: cycles max moyens ≈ {mean_c:.0f} (médiane {med_c:.0f})
        - Risques actuels (échelle interne): {crit} critiques, {warn} en alerte, {norm} normaux
        - Capacités: prédiction RUL, détection d'anomalies, segmentation et suivi en temps réel
        """.format(nb=nb_moteurs, mean_c=moyenne_cycles, med_c=mediane_cycles,
                   crit=int(counts.get('Critique', 0)), warn=int(counts.get('Alerte', 0)), norm=int(counts.get('Normal', 0))))

        st.markdown("### Recommandations Opérationnelles")
        st.write("""
        - Prioriser la maintenance préventive sur les moteurs en catégorie Critique (fenêtre < 10 cycles RUL lorsqu'estimée).
        - Surveiller de près les moteurs en Alerte et planifier une inspection lors du prochain créneau.
        - Capitaliser sur l'ingénierie des variables pour affiner les modèles (lissages, dérivées, agrégats temporels).
        - Envisager un modèle supervisé de classification du risque lorsque des labels seront disponibles.
        """)

        # Export synthèse
        summary_text = (
            "Synthèse Business AeroMaintain\n"
            "--------------------------------\n"
            f"Flotte analysée: {nb_moteurs} moteurs\n"
            f"Cycles max moyens: {moyenne_cycles:.0f} (médiane {mediane_cycles:.0f})\n"
            f"Répartition risques (échelle interne): Critique={int(counts.get('Critique', 0))}, "
            f"Alerte={int(counts.get('Alerte', 0))}, Normal={int(counts.get('Normal', 0))}\n\n"
            "Recommandations:\n"
            "- Maintenance prioritaire pour 'Critique'\n"
            "- Suivi renforcé pour 'Alerte'\n"
            "- Enrichir les variables et préparer labels pour un modèle supervisé\n"
        )

        st.download_button(
            label="📥 Télécharger la Synthèse (.txt)",
            data=summary_text,
            file_name="synthese_business_aeromaintain.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"⚠️ Erreur dans la Synthèse Business : {e}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #95a5a6; font-size: 12px;'>
    <p>🛩️ Tableau de Bord AeroMaintain | Dataset NASA C-MAPSS FD001 | Système de Maintenance Prédictive</p>
    <p>Construit avec Streamlit • Plotly • Scikit-learn • XGBoost</p>
</div>
""", unsafe_allow_html=True)
