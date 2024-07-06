export AIRFLOW_HOME=~/airflow
AIRFLOW_VERSION=2.1.0
PYTHON_VERSION=$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)
pip install -r requirement.txt

