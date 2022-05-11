install:venv
	. venv/bin/activate; python3 -m install -upgrade pip; pip3 install -Ur requirements.txt
venv:
	test -d venv || python3 -m venv venv
run:
	python3 Cluster.py
clean:
	rm -rf venv
	rm -rf Cluster.txt	