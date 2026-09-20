all:
	docker compose up --build -d

stop:
	docker compose down

train:
	docker compose up -d --build bike-api

evaluation:
	docker compose up -d --build evaluation

fire-alert:
	@echo "Déclenchement intentionnel de l'alerte HighModelRMSE (Prometheus) :"
	@echo "Envoi de données extrêmes à /evaluate pour forcer un RMSE anormalement élevé."
	curl -X POST http://localhost:8080/evaluate \
		-H "Content-Type: application/json" \
		-d '{"evaluation_period_name": "fire_alert_test", "data": [ \
			{"temp": 0.9, "atemp": 0.95, "hum": 0.05, "windspeed": 0.9, "mnth": 7, "hr": 12, "weekday": 3, "season": 3, "holiday": 0, "workingday": 1, "weathersit": 1, "cnt": 999, "dteday": "2011-07-15"}, \
			{"temp": 0.95, "atemp": 0.98, "hum": 0.02, "windspeed": 0.95, "mnth": 7, "hr": 13, "weekday": 3, "season": 3, "holiday": 0, "workingday": 1, "weathersit": 1, "cnt": 1000, "dteday": "2011-07-15"}, \
			{"temp": 0.02, "atemp": 0.01, "hum": 0.99, "windspeed": 0.02, "mnth": 1, "hr": 3, "weekday": 6, "season": 1, "holiday": 1, "workingday": 0, "weathersit": 4, "cnt": 1, "dteday": "2011-01-08"} ]}'
	@echo ""
	@echo "Attendez ~2 minutes (le 'for: 2m' de la règle HighModelRMSE) puis vérifiez http://<VM_IP>:9090/alerts"

.PHONY: all stop train evaluation fire-alert
