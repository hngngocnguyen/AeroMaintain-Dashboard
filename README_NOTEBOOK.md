# 🛩️ Dashboard Intelligent de Maintenance Prédictive Aéronautique
## Notebook Jupyter Complet - Projet M2 Data Science

---

## 📋 Vue d'Ensemble

Ce notebook implémente une **solution complète de maintenance prédictive** pour l'industrie aéronautique, utilisant le dataset NASA C-MAPSS et des techniques avancées de machine learning.

**Durée d'exécution estimée**: 1-2 heures  
**Langage**: Python 3.8+  
**Version du notebook**: 1.0

---

## 🎯 Objectifs

- ✅ Analyser **20,000+ observations** de capteurs moteur
- ✅ Identifier **patterns de dégradation** via clustering
- ✅ Prédire le **RUL (Remaining Useful Life)** avec R² > 0.8
- ✅ Classifier les moteurs par **niveau de risque**
- ✅ Générer insights pour un **dashboard interactif**

---

## 📦 Prérequis

### Python Packages
```bash
# Installation des dépendances
pip install pandas numpy scikit-learn xgboost plotly scipy

# Optionnel mais recommandé
pip install jupyter jupyterlab plotly-orca kaleido
```

### Données
Les fichiers de données doivent être dans le dossier `dataset/`:
```
dataset/
├── train_FD001.txt
├── train_FD002.txt
├── train_FD003.txt
├── train_FD004.txt
├── test_FD001.txt
├── test_FD002.txt
├── test_FD003.txt
├── test_FD004.txt
├── RUL_FD001.txt
├── RUL_FD002.txt
├── RUL_FD003.txt
└── RUL_FD004.txt
```

---

## 🗺️ Structure du Notebook (10 Sections)

### 1️⃣ **Initialisation et Préparation** 
- Import de toutes les bibliothèques
- Configuration Jupyter et Plotly
- Vérification des versions

### 2️⃣ **Chargement des Données NASA C-MAPSS**
- Lecture des 4 scénarios (FD001-FD004)
- Exploration de la structure (21 capteurs)
- Création de la variable cible (RUL)

### 3️⃣ **Exploration Avancée (EDA)**
- Distribution des cycles par scénario
- Matrice de corrélation entre capteurs
- Évolution temporelle de capteurs clés
- Identification des capteurs critiques

### 4️⃣ **Détection d'Anomalies**
- Détection Z-score
- Isolation Forest
- Score d'anomalie composite
- Visualisation des points aberrants

### 5️⃣ **Feature Engineering pour Séries Temporelles**
- Features glissantes (rolling statistics)
- Sélection de features via Mutual Information
- Normalisation StandardScaler
- Préparation des datasets train/test

### 6️⃣ **Clustering et Segmentation**
- PCA pour réduction dimensionnalité
- Elbow method pour k optimal
- K-Means clustering
- Analyse des profils par cluster

### 7️⃣ **Modélisation Prédictive du RUL**
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost (optionnel)
- Évaluation: MAE, RMSE, R²

### 8️⃣ **Classification du Risque**
- Création de classes (Sain/Dégradé/Critique)
- Random Forest Classifier
- Matrice de confusion
- Courbe ROC-AUC

### 9️⃣ **Évaluation Globale et KPIs**
- KPIs opérationnels (moteurs à risque, RUL par cluster)
- KPIs financiers (économies, ROI)
- Dashboard récapitulatif

### 🔟 **Synthèse Business et Recommandations**
- Résumé exécutif
- Insights stratégiques
- Plan d'implémentation
- Framework dashboard Dash

---

## 🚀 Comment Utiliser

### Option 1: Jupyter Notebook
```bash
# Lancer Jupyter
jupyter notebook Maintenance_Predictive_AeroMaintain.ipynb

# Ou avec JupyterLab
jupyter lab Maintenance_Predictive_AeroMaintain.ipynb
```

### Option 2: VS Code
1. Ouvrir le fichier `Maintenance_Predictive_AeroMaintain.ipynb`
2. VS Code détectera automatiquement le kernel Python
3. Exécuter les cellules avec `Shift+Enter`

### Option 3: Exécution complète
```bash
# Générer un rapport HTML
jupyter nbconvert --to html Maintenance_Predictive_AeroMaintain.ipynb
```

---

## 📊 Visualisations Générées

Le notebook crée **20+ graphiques interactifs** Plotly:

- 📈 Distribution des cycles par scénario (Box Plot)
- 🔥 Matrice de corrélation entre capteurs (Heatmap)
- 📉 Évolution temporelle des capteurs (Line Chart)
- 🚨 Détection d'anomalies (Scatter Plot)
- 🔀 Clustering (PCA 2D, clusters interactifs)
- 📊 Sélection features (Elbow method)
- 🎯 Comparaison modèles (MAE, RMSE, R²)
- 📉 Résidus prédiction (Histogramme, Scatter)
- 🔥 Matrice confusion (Heatmap)
- 📈 Courbe ROC (Line)
- 💰 Dashboard récapitulatif (KPIs)

**Tous les graphiques sont interactifs** (zoom, pan, hover, export)

---

## 📈 Résultats Attendus

### Performance du Modèle
- **R² Score**: > 0.85
- **MAE**: 10-15 cycles
- **Précision Classification**: > 85%
- **Recall Classification**: > 80%

### State of Fleet
- 🟢 **30-40%** moteurs sains
- 🟡 **40-50%** moteurs dégradés
- 🔴 **5-15%** moteurs critiques

### Impact Financier
- **Économies annuelles**: 500,000€+ (pour flotte de 150 moteurs)
- **ROI**: 300-400% en année 1
- **Réduction downtime**: 30-40%

---

## 🛠️ Fichiers Générés

Le notebook produit les fichiers suivants:

| Fichier | Description |
|---------|-------------|
| `predictions_moteurs.csv` | Prédictions RUL pour tous les moteurs |
| `SYNTHESE_EXECUTIVE.txt` | Résumé business en texte |
| `Maintenance_Predictive_AeroMaintain.html` | Rapport HTML exporté |

---

## 🎨 Palette de Couleurs

Pour la cohérence visuelle dans le dashboard:

```python
COLOR_PALETTE = {
    'primary': '#3498db',      # Bleu
    'secondary': '#2ecc71',    # Vert
    'warning': '#f39c12',      # Orange
    'danger': '#e74c3c',       # Rouge
    'neutral': '#95a5a6',      # Gris
    'dark': '#2c3e50'          # Bleu foncé
}
```

---

## 📝 Notes Techniques

### Hyperparamètres Clés
- **Random Forest**: 100 trees, max_depth=20
- **Gradient Boosting**: learning_rate=0.1, n_estimators=100
- **XGBoost**: max_depth=5, learning_rate=0.1
- **K-Means**: optimal k déterminé par silhouette score

### Seuils RUL
- 🟢 **Sain**: RUL > 30 cycles
- 🟡 **Dégradé**: 10 < RUL ≤ 30 cycles  
- 🔴 **Critique**: RUL ≤ 10 cycles

### Contamination Anomalies
- **Isolation Forest**: contamination = 5%
- **Z-score**: seuil = 3σ

---

## 🐛 Troubleshooting

### Issue: Erreur lors du chargement des données
```
FileNotFoundError: dataset/train_FD001.txt not found
```
**Solution**: Vérifier que le dossier `dataset/` existe et contient les 12 fichiers txt

### Issue: XGBoost non disponible
```
ImportError: No module named 'xgboost'
```
**Solution**: `pip install xgboost` (optionnel, le notebook fonctionne sans)

### Issue: Plots Plotly ne s'affichent pas
**Solution**: Mettre à jour Plotly: `pip install --upgrade plotly`

---

## 📚 Ressources Complémentaires

### Documentation
- 📖 [Plotly Documentation](https://plotly.com/python/)
- 📖 [Scikit-learn Guide](https://scikit-learn.org/stable/)
- 📖 [NASA C-MAPSS](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

### Articles Références
- "Predictive Maintenance Using Machine Learning" - IEEE
- "Feature Engineering for Predictive Maintenance" - ACM
- "Deep Learning for Time Series Forecasting" - arXiv

---

## 🎯 Prochaines Étapes

1. **Déployer le dashboard Dash** (`dashboard_aeromaintain.py`)
2. **Intégrer données temps réel** depuis capteurs IoT
3. **Implémenter LSTM** pour séries longues
4. **Ajouter explainability** avec SHAP/LIME
5. **Mettre en production** avec API REST

---

## 📞 Support

**Responsable Projet**: Équipe Data Science  
**Email**: data-science@aeromaintain.fr  
**Version Notebook**: 1.0  
**Dernière mise à jour**: 2025  
**License**: Propriétaire AeroMaintain Solutions

---

## ✨ Crédits

Créé pour **AeroMaintain Solutions** dans le cadre d'un **projet M2 Data Science**.

Dataset: NASA Prognostics Data Repository (C-MAPSS)

**Bon travail! 🚀**
