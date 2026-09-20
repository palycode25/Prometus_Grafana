# Bike Sharing MLOps Monitoring — Rendu d'examen

## 1. Objectif du projet

Ce projet met en place une solution complète de monitoring MLOps pour un modèle de
régression prédisant le nombre de vélos partagés (`cnt`) à partir du dataset
**Bike Sharing UCI**. L'objectif est de surveiller en continu :
- la qualité du modèle (RMSE, MAE, R², MAPE)
- la dérive des données (data drift) via Evidently
- la santé de l'API (latence, taux d'erreur)
- l'infrastructure hôte (CPU, RAM, disque)

## 2. Architecture

Services Docker Compose :
- bike-api (FastAPI, port 8080) : prédiction + évaluation + métriques
- prometheus (port 9090) : collecte et stockage des métriques (TSDB)
- grafana (port 3000) : visualisation, dashboards, alerting
- node-exporter (port 9100) : métriques système de la machine hôte
- evaluation : script run_evaluation.py (batch, à la demande)

## 3. Modèle de Machine Learning

- Algorithme : RandomForestRegressor (scikit-learn)
- Cible : cnt (nombre de vélos loués)
- Features numériques : temp, atemp, hum, windspeed, mnth, hr, weekday
- Features catégorielles : season, holiday, workingday, weathersit
- Données de référence (entraînement) : janvier 2011 (688 échantillons horaires)
- Entraînement : au démarrage du conteneur bike-api, une seule fois, modèle
  gardé en mémoire pour l'inférence

## 4. Endpoints de l'API (src/api/main.py)

- GET / : message de bienvenue
- POST /predict : prend BikeSharingInput, retourne la prédiction du nombre de vélos
- POST /evaluate : prend un lot de données avec la vraie valeur cnt, calcule les
  prédictions, exécute un rapport Evidently comparant au jeu de référence (janvier
  2011), extrait RMSE/MAE/R2/MAPE/drift, met à jour Prometheus, renvoie un résumé
- GET /metrics : expose toutes les métriques au format Prometheus

## 5. Métriques Prometheus exposées

- api_requests_total (Counter) : nb total de requêtes, labels endpoint/method/status_code
- api_request_duration_seconds (Histogram) : latence des requêtes API
- model_rmse_score (Gauge) : RMSE du modèle sur le dernier lot évalué
- model_mae_score (Gauge) : MAE du modèle sur le dernier lot évalué
- model_r2_score (Gauge) : R2 du modèle sur le dernier lot évalué
- model_mape_score (Gauge) : MAPE du modèle sur le dernier lot évalué
- model_data_drift_score (Gauge) : METRIQUE PERSONNALISEE, part des features driftées (0-1)

### Justification de la métrique personnalisée : model_data_drift_score

RMSE/MAE/R2/MAPE mesurent la qualité des prédictions a posteriori, mais ne disent
pas pourquoi le modèle se dégrade. Le score de dérive des données (calculé via
DataDriftPreset d'Evidently sur les features d'entrée : météo, saison, etc.) permet
d'anticiper une dégradation avant même que les métriques de qualité ne se détériorent
fortement. C'est un signal d'alerte précoce qui guide la décision de ré-entraînement,
particulièrement pertinent pour un dataset saisonnier comme Bike Sharing.

Le calcul utilise une fonction robuste _recursive_find_metric() qui parcourt la
structure JSON retournée par Evidently pour extraire dynamiquement les valeurs
(certaines métriques renvoient un float direct, d'autres un dict {"mean":...,"std":...}
selon le type). Une fonction _sanitize_float() protège aussi contre les NaN/Inf qui
peuvent apparaître avec de très petits lots de données (ex: R2 indéfini avec 1 seul
point), évitant des erreurs de sérialisation JSON ou des valeurs invalides côté
Prometheus.

## 6. Alerting

### Alertes Prometheus (deployment/prometheus/rules/alert_rules.yml)
- BikeApiDown : up{job="bike_api"} == 0, pending 1m, sévérité critical
- HighModelRMSE : model_rmse_score > 150, pending 2m, sévérité warning

### Alerte Grafana
- ModelDriftHigh : model_data_drift_score > 0.3, pending 1m, sévérité warning
  Configurée via l'UI Grafana (Alerting > Alert rules), notification routée vers
  le contact point "default-contact" (email factice, suffisant pour la démo).

### Test de l'alerte : make fire-alert

La cible fire-alert envoie volontairement des données extrêmes (température très
élevée en plein hiver, humidité quasi nulle, etc.) à /evaluate. Ces valeurs
aberrantes par rapport à la référence de janvier 2011 poussent le RMSE au-dessus
du seuil de 150, ce qui déclenche l'alerte Prometheus HighModelRMSE après la
période pending de 2 minutes (visible sur http://<IP_VM>:9090/alerts).

## 7. Dashboards Grafana (Dashboards as Code)

Les 3 dashboards sont définis en JSON dans deployment/grafana/dashboards/ et
chargés automatiquement au démarrage de Grafana via le provisioning
(deployment/grafana/provisioning/), sans aucune configuration manuelle.

- API Performance : taux de requêtes par endpoint, latence P95, taux d'erreur,
  nombre total de requêtes.
- Model Performance & Drift : RMSE/MAE/R2/MAPE, score de dérive des données avec
  seuils visuels (vert/jaune/rouge), évolution temporelle des scores.
- Infrastructure Overview : usage CPU, RAM, disque (via node-exporter).

La datasource Prometheus et le provider de dashboards sont provisionnés
automatiquement — testé et validé sur relance complète du projet (docker compose
down puis docker compose up -d --build).

## 8. Makefile

- make all : démarre tous les services (bike-api, prometheus, grafana, node-exporter)
- make stop : arrête tous les services
- make evaluation : exécute run_evaluation.py (envoie un batch réel à /evaluate +
  génère 50 requêtes /predict)
- make train : relance/rebuild bike-api (ré-entraîne le modèle au redémarrage)
- make fire-alert : déclenche intentionnellement l'alerte Prometheus HighModelRMSE

## 9. Simulation de trafic

Le script src/evaluation/run_evaluation.py effectue deux actions à chaque
exécution (make evaluation) :
1. Évaluation : envoie un échantillon réel de la semaine 1 de février 2011
   (235 lignes) à /evaluate, met à jour toutes les métriques ML dans Prometheus.
2. Génération de trafic : envoie 50 requêtes /predict avec des données réelles
   de janvier 2011, simulant une utilisation normale de l'API.

## 10. Comment lancer le projet (reproductibilité)

    git clone <repo>
    cd PromGraf-MLOps-Exam-Student
    make all

Attendre environ 1 minute que bike-api télécharge les données UCI et entraîne
le modèle. Puis :
- API : http://<IP_VM>:8080
- Prometheus : http://<IP_VM>:9090
- Grafana : http://<IP_VM>:3000 (login admin/admin, dashboards et datasource
  déjà configurés automatiquement)

Pour générer des données de test : make evaluation
Pour tester l'alerting : make fire-alert (puis attendre ~2 min et vérifier
http://<IP_VM>:9090/alerts)

## 11. Choix techniques notables

- Evidently 0.7.0 (nouvelle API Dataset/DataDefinition/Report) plutôt que
  l'ancienne API, conformément aux imports fournis dans le squelette de départ.
- CollectorRegistry personnalisé plutôt que le registre global de
  prometheus_client, pour n'exposer que les métriques pertinentes au projet.
- Gestion défensive des NaN/Inf avant sérialisation JSON et mise à jour des
  Gauges Prometheus, pour éviter tout crash sur des cas limites.
