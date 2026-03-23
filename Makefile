setup:
	docker compose build
	docker compose up -d

up:
	docker compose up -d

build:
	docker compose build

restart:
	docker compose restart

test:
	docker compose exec core pytest

down:
	docker compose down

logs:
	docker compose logs -f