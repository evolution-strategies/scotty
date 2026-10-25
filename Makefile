.PHONY: verify test results paper

verify:
	python3 tools/verify_repository.py

test:
	python3 -m unittest discover -s tests -v

results:
	python3 tools/show_results.py

paper:
	cd paper/source && ./build.sh
	cp paper/source/scotty1.pdf paper/Scotty-1_ESANN_paper.pdf
