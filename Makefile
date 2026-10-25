.PHONY: verify test results

verify:
	python3 tools/verify_repository.py

test:
	python3 -m unittest discover -s tests -v

results:
	python3 tools/show_results.py
