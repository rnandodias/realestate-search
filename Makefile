.PHONY: up down logs ps
up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

ps:
	docker ps --format '{{.Names}} -> {{.Status}}'