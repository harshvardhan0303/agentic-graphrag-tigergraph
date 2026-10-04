.PHONY: setup doctor models ingest vectors oracle deploy bench-public bench-hidden dashboard test all

setup:
	pip install -r requirements.txt
	cp -n .env.example .env || true

doctor:
	python scripts/doctor.py

models:
	python scripts/list_models.py

ingest:
	python -m src.ingest.parse_corpus --corpus data/corpus.jsonl --out data/graph

vectors:
	python -m src.vectorstore.build --chunks data/graph/chunks.jsonl --out data/vectors

oracle:
	python -m src.bench.oracle --questions data/eval_public.jsonl

deploy:
	python -m src.graph.deploy --all

bench-public:
	python -m src.bench.run --questions data/eval_public.jsonl --out out/public --ablation

bench-hidden:
	python -m src.bench.run --questions data/eval_hidden.jsonl --out out/hidden --no-grade

dashboard:
	python -m src.bench.dashboard --run out/public --out out/dashboard.html

test:
	pytest -q

all: ingest vectors oracle test
