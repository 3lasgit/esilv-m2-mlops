# 🫀 Heart Disease Prediction — MLOps Final Project

> **Cours** : Machine Learning & MLOps
> **Encadrants** : yue.li@ext.devinci.fr · linghao.kong@ext.devinci.fr
> **Deadline** : 15 mars 2026

---

## 1. Description

Ce projet applique un pipeline MLOps complet à la prédiction de maladies cardiaques à partir de données cliniques. L'objectif est de produire un modèle de classification binaire (présence ou absence de maladie cardiaque) reproductible, versionné, déployable et monitoré, en suivant les bonnes pratiques de l'ingénierie ML en production.

Le projet couvre l'ensemble de la chaîne : exploration des données, feature engineering, entraînement de modèles classiques et d'un réseau de neurones, experiment tracking avec MLflow, exposition via une API REST, containerisation Docker, et monitoring en production.

---

## 2. Dataset — Heart Disease UCI (Cleveland)

| Propriété | Valeur |
|-----------|--------|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Origine | Cleveland Clinic Foundation (1988) |
| Observations | 303 patients |
| Features | 13 variables cliniques |
| Cible | `target` — 0 = absence, 1 = présence de maladie cardiaque |
| Équilibre | ~54 % positifs / 46 % négatifs |

### Variables

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numérique | Âge du patient (années) |
| `sex` | Binaire | Sexe (1 = homme, 0 = femme) |
| `cp` | Catégorielle (0–3) | Type de douleur thoracique |
| `trestbps` | Numérique | Pression artérielle au repos (mmHg) |
| `chol` | Numérique | Cholestérol sérique (mg/dl) |
| `fbs` | Binaire | Glycémie à jeun > 120 mg/dl |
| `restecg` | Catégorielle (0–2) | Résultats ECG au repos |
| `thalach` | Numérique | Fréquence cardiaque maximale atteinte |
| `exang` | Binaire | Angine induite par l'effort |
| `oldpeak` | Numérique | Dépression du segment ST à l'effort |
| `slope` | Catégorielle (0–2) | Pente du segment ST |
| `ca` | Numérique (0–3) | Nombre de vaisseaux colorés (fluoroscopie) |
| `thal` | Catégorielle (0–2) | Thalassémie |

### Features dérivées (feature engineering)

Trois variables d'interaction à pertinence clinique ont été créées :

| Feature | Formule | Interprétation |
|---------|---------|----------------|
| `hr_age_ratio` | `thalach / age` | Réserve cardiaque normalisée par l'âge |
| `cardio_risk_score` | `(trestbps/120) × (chol/200) × (age/60)` | Score de risque cardiovasculaire composé |
| `exang_oldpeak` | `exang × oldpeak` | Stress cardiaque combiné |

---

## 3. Task Definition & Résultats

### Problème

**Classification binaire supervisée** : prédire si un patient présente une maladie cardiaque (`target = 1`) ou non (`target = 0`) à partir de ses données cliniques et biologiques.

### Métrique principale

**AUC-ROC** — choisie car elle est insensible au seuil de décision et adaptée aux applications médicales où l'équilibre précision/rappel est critique.
Métriques secondaires : F1-score, Accuracy.

### Modèles entraînés

| Modèle | Type | Tuning |
|--------|------|--------|
| Logistic Regression | Linéaire | — |
| K-Nearest Neighbors | Instance-based | Sélection du k optimal (CV) |
| Decision Tree | Arbre | — |
| Random Forest | Ensemble | GridSearchCV |
| SVM (RBF) | Noyau | RandomizedSearchCV |
| ANN | Deep Learning | Early Stopping + ReduceLROnPlateau |

### Résultats clés (test set)

| Modèle | AUC-ROC | F1-Score | Accuracy |
|--------|---------|----------|----------|
| RF (Tuned) | **0.9012** | **0.8649** | — |
| SVM (Tuned) | 0.9032 | — | — |
| KNN | 0.8918 | — | — |
| ANN | ~0.84 | — | — |

> Les valeurs exactes sont loggées dans MLflow lors de chaque run d'entraînement (`python src/train.py`).

---

## 4. Architecture & Stack technique

### Stack

| Composant | Outil |
|-----------|-------|
| Environnement | [uv](https://github.com/astral-sh/uv) |
| Qualité de code | ruff · black · isort · mypy · pre-commit |
| ML | scikit-learn · TensorFlow/Keras · SHAP |
| Experiment tracking | MLflow (tracking + model registry) |
| API | FastAPI + Pydantic |
| Containerisation | Docker · docker-compose |
| CI/CD | GitHub Actions (lint + test + build + smoke test) |
| Tests | pytest (≥ 60 % coverage, atteint 91 %) |
| Monitoring | `/health` + `/metrics` + logging structuré JSON |

### Structure du projet

```
.
├── src/
│   ├── data.py              # Chargement, preprocessing, feature engineering
│   ├── model.py             # Définition des modèles, tuning, ANN
│   ├── train.py             # Script d'entraînement + logging MLflow
│   ├── registry.py          # MLflow Model Registry (promote, compare)
│   └── logging_config.py    # Logging structuré JSON (production)
├── api/
│   └── app.py               # FastAPI — /predict, /health, /metrics
├── tests/
│   ├── test_data.py         # Tests preprocessing
│   ├── test_model.py        # Tests modèles
│   ├── test_api.py          # Tests endpoints API
│   ├── test_mlflow.py       # Tests logging MLflow
│   └── test_monitoring.py   # Tests monitoring + structured logging
├── models/                  # Artefacts sauvegardés (.gitignore)
├── reports/
│   └── README.md            # Lien vidéo démo
├── .github/workflows/
│   ├── ci.yml               # Pipeline CI — lint + pytest
│   └── cd.yml               # Pipeline CD — Docker build + smoke test
├── Dockerfile.train         # Image d'entraînement
├── Dockerfile.inference     # Image API (inférence)
├── docker-compose.yml
├── pyproject.toml           # Dépendances gérées avec uv
└── README.md
```

### Démarrage rapide

```bash
# 1. Cloner le dépôt
git clone https://github.com/3lasgit/esilv-m2-mlops.git && cd esilv-m2-mlops

# 2. Installer l'environnement avec uv
uv sync

# 3. Lancer l'entraînement (avec MLflow tracking)
uv run python src/train.py

# 4. Lancer l'API
uv run uvicorn api.app:app --reload

# 5. Tester un patient
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":54,"sex":1,"trestbps":130,"chol":256,"fbs":0,"thalach":147,"exang":0,"oldpeak":1.4,"ca":0,"cp_0":1,"cp_1":0,"cp_2":0,"cp_3":0,"restecg_0":1,"restecg_1":0,"restecg_2":0,"slope_0":0,"slope_1":1,"slope_2":0,"thal_0":0,"thal_1":0,"thal_2":1,"hr_age_ratio":2.72,"cardio_risk_score":1.30,"exang_oldpeak":0.0}'
```

### Docker

```bash
# Entraînement
docker build -f Dockerfile.train -t heart-train .
docker run heart-train

# API
docker build -f Dockerfile.inference -t heart-api .
docker run -p 8000:8000 -e USE_LOCAL_MODEL=true heart-api
```

---

## 5. MLOps Practices

### 5.1 Code Quality & Pre-commit

Chaque commit passe automatiquement par 4 hooks via `pre-commit` :

- **ruff** — linting rapide (erreurs, imports, simplifications)
- **black** — formatage automatique (line-length 100)
- **isort** — tri des imports (profil black)
- **mypy** — vérification de types statiques

Configuration centralisée dans `pyproject.toml`.

### 5.2 Tests & Coverage

Les tests unitaires couvrent l'ensemble de la codebase :

| Module testé | Fichier de test | Contenu |
|-------------|-----------------|---------|
| `src/data.py` | `tests/test_data.py` | Preprocessing, outliers, encoding, feature engineering |
| `src/model.py` | `tests/test_model.py` | Modèles baseline, tuning RF/SVM, architecture ANN |
| `api/app.py` | `tests/test_api.py` | Endpoints /predict, /health, validation Pydantic, erreurs 422/503 |
| `src/train.py` | `tests/test_mlflow.py` | Logging MLflow (params, metrics, artifacts) |
| Monitoring | `tests/test_monitoring.py` | Endpoints /metrics, structured logging JSON |

**Coverage atteint : 91 %** (seuil minimal CI : 60 %).

### 5.3 Experiment Tracking — MLflow

Le script `src/train.py` intègre MLflow nativement :

- **Expérience** : `heart-disease-prediction`
- **Run parent** `full_training` contient les paramètres globaux (random_state, cv_splits, n_features)
- **Nested runs** pour chaque modèle (baseline + tuned) avec params, métriques (cv_accuracy, cv_f1, cv_auc, test_accuracy, test_f1, test_auc), et artifact du modèle
- **Model Registry** via `src/registry.py` : promotion du meilleur run en Staging → Production
- **Best metric** : `best_test_auc` loggée sur le run parent pour comparaison rapide

```bash
# Comparer les runs
uv run python src/registry.py --compare

# Promouvoir le meilleur modèle
uv run python src/registry.py --promote

# Passer en Production
uv run python src/registry.py --to-prod
```

### 5.4 API REST — FastAPI

L'API expose 3 endpoints :

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/predict` | POST | Prédiction binaire (maladie cardiaque) |
| `/health` | GET | Statut de l'API, version du modèle, uptime |
| `/metrics` | GET | Compteurs de prédictions, latences (avg, p95), erreurs |

Le modèle est chargé au démarrage depuis **MLflow Model Registry** (fallback : fichier local). Les features sont normalisées via le scaler sauvegardé. Validation stricte des entrées via Pydantic (bornes, types).

### 5.5 Containerisation — Docker

Deux images Docker distinctes :

- **`Dockerfile.train`** — Exécute `src/train.py` (entraînement complet + logging MLflow)
- **`Dockerfile.inference`** — Sert l'API FastAPI sur le port 8000 avec healthcheck intégré

Les deux images utilisent `uv` pour la résolution de dépendances (lockfile frozen). L'image d'inférence tourne sous un utilisateur non-root (`appuser`).

### 5.6 CI/CD — GitHub Actions

**Pipeline CI** (`.github/workflows/ci.yml`) — déclenché sur push et PR vers `main` :
1. **Job lint** : ruff check, black --check, isort --check, mypy
2. **Job test** : pytest avec coverage ≥ 60 %

**Pipeline CD** (`.github/workflows/cd.yml`) — déclenché sur push vers `main` :
1. **Build & push** des images Docker (train + inference) vers GHCR avec tags `latest` + `sha`
2. **Smoke test** : lance le container API et vérifie que `/health` répond

---

## 6. Monitoring & Observabilité

### Endpoint `/health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "Staging",
  "model_source": "mlflow:http://localhost:5000",
  "uptime_seconds": 1234.5
}
```

Utilisé par le `HEALTHCHECK` Docker et le smoke test CD.

### Endpoint `/metrics`

```json
{
  "total_predictions": 142,
  "predictions_positive": 67,
  "predictions_negative": 75,
  "errors_count": 0,
  "avg_latency_ms": 3.21,
  "p95_latency_ms": 8.45,
  "uptime_seconds": 3600.0
}
```

Permet de détecter la dérive de distribution des prédictions (ratio positif/négatif), les dégradations de performance (latence), et les erreurs.

### Logging structuré (JSON)

En production (`APP_ENV=production`), tous les logs sont émis en JSON structuré, compatible avec les stacks d'observabilité (ELK, CloudWatch, Datadog) :

```json
{
  "timestamp": "2026-03-15T10:30:00+00:00",
  "level": "INFO",
  "logger": "api",
  "message": "Prediction effectuee",
  "prediction": 1,
  "probability": 0.8234,
  "latency_ms": 3.45,
  "age": 54.0
}
```

En développement (`APP_ENV=dev`), les logs sont formatés de manière lisible pour la console.

---

## 7. Répartition des rôles & Collaboration

### Organisation Git

Le projet suit un workflow **feature-branch** avec revue croisée systématique :

| PR | Branche | Responsable | Contenu |
|----|---------|-------------|---------|
| #1 | `feature/project-structure` | A | Init uv, refactoring notebook → `src/`, README initial |
| #2 | `feature/code-quality` | B | Pre-commit hooks, tests unitaires (≥ 60 %), CI lint/pytest |
| #3 | `feature/mlflow_and_docker` | A+B | MLflow tracking + model registry, FastAPI, Dockerfiles |
| #4 | `feature/cicd-monitoring-report` | B | Pipeline CD, `/metrics`, logging structuré, README final, vidéo |

Chaque PR est revue par le binôme avant merge sur `main`.

### Vidéo démo

Le lien vers la vidéo de démonstration (pipeline CI/CD + API) est disponible dans [`reports/README.md`](reports/README.md).

---

## 8. Limitations & Perspectives

### Limitations actuelles

- **Taille du dataset** — 303 observations, insuffisant pour un déploiement clinique réel. Les performances sont encourageantes mais manquent de robustesse statistique.
- **Monitoring simplifié** — Les métriques sont stockées en mémoire (reset au redémarrage). Une solution persistante (Prometheus + Grafana) serait nécessaire en production.
- **Pas de détection de drift** — Aucun mécanisme automatique ne surveille la dérive des données d'entrée ou des prédictions par rapport à la distribution d'entraînement.
- **Modèle unique en production** — Seul le Random Forest tuné est servi. Un système d'A/B testing ou de shadow deployment permettrait de comparer les modèles en conditions réelles.
- **Features calculées côté client** — Les features dérivées (`hr_age_ratio`, `cardio_risk_score`, `exang_oldpeak`) doivent être calculées avant l'appel API. Idéalement, l'API recevrait les 13 features brutes et calculerait les dérivées.

### Pistes d'amélioration

- **Data augmentation** — Enrichir le dataset avec d'autres sources (Framingham, Kaggle) ou utiliser des techniques de génération synthétique (SMOTE, CTGAN).
- **Monitoring avancé** — Intégrer Prometheus pour l'export des métriques, Grafana pour les dashboards, et Evidently AI pour la détection automatique de data/concept drift.
- **Feature store** — Centraliser le calcul et le versionnement des features via Feast ou un feature store maison.
- **CI/CD complet** — Ajouter un stage de retraining automatique si la performance dégrade, et un déploiement blue/green sur Kubernetes.
- **Explicabilité** — Exposer les SHAP values via un endpoint `/explain` pour chaque prédiction individuelle.

---

## Auteurs
|------|---------|
| BLIDI Ala |  |
| TLEMSANI Sofiane | — |
