Machine Learning Project – Part 1 and Part 2

This project contains the implementation for Part 1 and Part 2 of a regression-based
Machine Learning assignment using Python.

--------------------------------------------------
CREATE VIRTUAL ENVIRONMENT
--------------------------------------------------

Windows:
python -m venv venv
venv\Scripts\activate

macOS / Linux:
python3 -m venv venv
source venv/bin/activate

--------------------------------------------------
INSTALL REQUIREMENTS
--------------------------------------------------

All required libraries are listed in requirements.txt.

pip install -r requirements.txt

--------------------------------------------------
RUN THE PROJECT
--------------------------------------------------

Activate the virtual environment and run the required part from the project root.

Run Part 1:
python src/part_1.py

Run Part 2:
python src/part_2.py

RUN PART 2 EVALUATION

Execute the official evaluation using the provided framework:

python framework_58.py --eval_file_path problem_58/EVAL_58.csv

--------------------------------------------------
IMPORTANT FILES
--------------------------------------------------

src/part_1.py     -> Part 1 model training and evaluation  
src/part_2.py     -> Part 2 model training and evaluation  
framework_58.py   -> predictions
requirements.txt  -> Dependency list  
README.md         -> Project instructions  

--------------------------------------------------
NOTES
--------------------------------------------------

- Ensure dataset paths inside the scripts are correct.
- Activate the virtual environment before running the code.
- Use Python 3.9 or higher.