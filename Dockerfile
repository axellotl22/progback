FROM ubuntu

RUN apt-get update
RUN apt-get -y install python3 python3-pip

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY ./app/ .

# Start API
ENTRYPOINT ["python3", "main.py"]