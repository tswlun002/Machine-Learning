install:venv
	. venv/bin/activate; python3 -m install -upgrade pip;mkdir Models; pip3 install -Ur requirements.txt
venv:
	test -d venv || python3 -m venv venv
run1:
	python3 Classifier_1.py
run2:
	python3 Classifier_2.py
run3:
	python3 Classifier_3.py
run4:
	python3 Classifier_4.py
run5:
	python3 Classifier_5.py
run6:
	python3 Classifier_6.py
clean:
	rm -rf venv
	rm -rf Models