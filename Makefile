.PHONY: install mock test verify verify-real run stream trace demo docker clean

PY ?= python3
MOCK_URL = http://127.0.0.1:8008/v1/

install:            ## create venv and install with dev extras
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -q --upgrade pip && pip install -q -e ".[dev]"

mock:               ## run the mock Token Factory on :8008 (foreground)
	. .venv/bin/activate && uvicorn mock_server.server:app --port 8008

test:               ## offline test-suite (starts its own mock)
	. .venv/bin/activate && python -m pytest

verify:             ## conformance kit against the mock (needs `make mock` in another shell)
	. .venv/bin/activate && NEBIUS_BASE_URL=$(MOCK_URL) NEBIUS_API_KEY=mock TRACE_EXPORTER=none nebius-agent verify --json reports/mock.json

verify-real:        ## conformance kit against Token Factory (needs NEBIUS_API_KEY, NEBIUS_MODEL)
	. .venv/bin/activate && nebius-agent verify --json reports/$$(echo $${NEBIUS_MODEL:-default} | tr '/' '_').json

run:                ## one agent run with structured output + trace file
	. .venv/bin/activate && nebius-agent run "$(PROMPT)"

stream:             ## streaming text answer
	. .venv/bin/activate && nebius-agent stream "$(PROMPT)"

trace:              ## summarise the last trace file
	. .venv/bin/activate && nebius-agent trace

demo:               ## the panel demo script
	bash scripts/demo.sh

docker:             ## build the service image
	docker build -t nebius-latency-scout:0.1.0 .

clean:
	rm -rf .venv traces reports/*.json .pytest_cache src/*.egg-info
