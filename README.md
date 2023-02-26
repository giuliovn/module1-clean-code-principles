# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The purpose of this project is to predict customer churn rate for credit card holder.

## Project Setup
- [Install poetry](https://python-poetry.org/docs/)
- Install requirements with either:
  - `poetry install`
  - export to requirements.txt with `poetry export --without-hashes -f requirements.txt -o requirements.txt` and `pip install -r requirements.txt`
### Docker image


## Files and data description
The data set, for convenience, is saved in `data` directory
All the outputs will be written to `outputs` directory, unless differently specified with command line argument `-o | --output`

## Running Files
### Python
**NB** if running using poetry prepend `poetry run` to any python command
```shell
python churn_library.py data/bank_data.csv (-o /path/to/outputs)
```
### Docker
#### Build
`docker build . -t churn`
#### Run
`docker run -v /path/to/output:/app/churn/outputs -v $(pwd)/data:/data churn /data/bank_data.csv`
**WARNING** docker user is root


