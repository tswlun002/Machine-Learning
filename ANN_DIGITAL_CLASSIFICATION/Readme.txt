Build virtual environment command:  $ make
Activate virtual environment: $ source ./venv/bin/activate

before run make sure MNIST is this current directory with source files and unzipped
Please don't delete folder Models , i made to save time by saving models to it so that you don't retrain model
when there is mistake while test the assignment . This folder will be deleted when you call $ make clean at the end.

executing code command : $ make run1   or $ python3   classifier_1.py          # for classifier_1.py
                        $ make run2   or $ python3   classifier_2.py          # for classifier_2.py
                        $ make run3   or $ python3   classifier_3.py          # for classifier_3.py
                        $ make run4   or $ python3   classifier_4.py          # for classifier_4.py
                        $ make run5   or $ python3   classifier_5.py          # for classifier_5.py
                        $ make run6   or $ python3   classifier_6.py          # for classifier_6.py

clean command: $ make clean        # venv and Models folder will be deleted