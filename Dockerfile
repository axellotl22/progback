FROM python:3.10-slim

ARG DEV_MODE=True
ENV DEV_MODE ${DEV_MODE}
ENV TEST_MODE False

ARG D_DB_HOST=""
ARG D_DB_PORT=""
ARG D_DB_PW=""
ARG D_DB_SCHEMA=""
ENV D_DB_HOST ${D_DB_HOST}
ENV D_DB_PORT ${D_DB_PORT}
ENV D_DB_PW ${D_DB_PW}
ENV D_DB_SCHEMA ${D_DB_SCHEMA}
ENV APP_SECRET ${APP_SECRET}
ENV VERIFICATION_SECRET ${VERIFICATION_SECRET}

# Setzen der Umgebungsvariablen, um Python davon abzuhalten, .pyc-Dateien im Container zu generieren
# und das Puffering zu deaktivieren, um das Container-Logging zu erleichtern
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Setzen des Arbeitsverzeichnis im Container
WORKDIR /code

# Installieren der Systemabhängigkeiten
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

# Kopieren des aktuellen Verzeichnisinhalte in das Container-Verzeichnis /code
COPY ./requirements.txt /code/
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Kopieren des App-Code in den Container
COPY ./app /code/app

# Starten Sie die API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
