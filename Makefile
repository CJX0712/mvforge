.PHONY: test demo lock lint fmt

test:
	pytest -q -W ignore::UserWarning

demo:
	python -m mvforge.examples.run_demo

lock:
	pip freeze > requirements.lock.txt

lint:
	ruff check .

fmt:
	ruff format --check .
