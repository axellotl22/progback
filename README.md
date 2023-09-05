# Programmierprojekt Backend K-means | Decision Trees

This repository contains a minimum RESTful API using Python and Flask. It exposes the endpoint http://HOST:5000/hello. This repository uses a Github Flow branching strategy and Github Actions to run _pylint_ to lint the code and _pytest_ to execute unit tests.

## Local API setup

Running the API locally requires Python 3.9, install that first. Then:

``` bash
python3 -m venv progback # create virtual environment
source progback/bin/activate # activate virtual environment
pip3 install -r requirements.txt # install python packages

uvicorn app.main:app --reload # start API
```

Now, you can access _http://127.0.0.1:8080/docs

## Local Docker execution

Building and running Docker containers requires Docker to be installed. Then:
``` bash
docker build -t progback . # build Docker container
docker run -it -p 8080:8080 progback # run Docker container
```

Now, you can access _http://127.0.0.1:8080/docs
