install:venv
	. venv/bin/activate; python3 -m install -upgrade pip;mkdir "Images";pip3 install -Ur requirements.txt
venv:
	test -d venv || python3 -m venv venv
run1:
	python3 Scenario1.py
clean:
	rm -rf venv
	rm -rf Images
