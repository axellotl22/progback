# Clustering-API

Das Projekt stellt eine API bereit, mit der Datenpunkte mithilfe des Clustering-Algorithmus gruppiert werden können. Benutzer können ihre Daten in Form von Excel-Dateien hochladen, und die API gibt die geclusterten Datenpunkte zurück.

## Inhaltsverzeichnis

- [Voraussetzungen](#Voraussetzungen)
- [Installation und Einrichtung](#installation-und-einrichtung)
- [Repository-Struktur](#repository-struktur)
- [Deployment](#deployment)
- [API-Dokumentation](#api-dokumentation)

## Voraussetzungen

- Docker und Docker Compose: Zum Containerisieren der Anwendung.
- LazyDocker: Ein einfacher Terminal-UI für Docker (wird in der ./deploy.sh mitgeliefert).

**Hinweis für Windows-Nutzer:** Die Anwendung funktioniert nur unter Windows Subsystem for Linux (WSL). Sie können WSL mithilfe der [offiziellen Dokumentation](https://docs.microsoft.com/de-de/windows/wsl/install) von Microsoft installieren und einrichten.

## Installation und Einrichtung

```bash
git clone https://github.com/axellotl22/progback
cd progback
```
## Repository-Struktur
```bash
Progback/
│
├── app/                        # Hauptanwendungsverzeichnis
│ ├── routers/                  # FastAPI-Endpunkte
│ │ ├── clustering_router.py    # Endpunkt für das Hochladen und Clustern von Dateien
│ ├── models/                   # Datenmodelle und -schemata
│ │ ├── clustering_model.py     # Modelle für Eingabe-/Ausgabedaten
│ ├── services/                 # Dienstprogramme und Services
│ │ ├── clustering_service.py   # Dienstprogramme für KMeans-Clustering
│ └── main.py                   # Hauptanwendungsdatei
│
├── temp_files/                 # Verzeichnis für hochgeladene Dateien 
|
├── tests/                      # Testverzeichnis
│ ├── __init__.py
│ ├── test_app.py               # Haupttestdatei
│
├── .github/                    # GitHub-spezifische Dateien
│ └── workflows/                # CI/CD-Workflows
│
├── deployment.sh               # Automatisierte Bereitstellung des Containers und Lazydocker
├── docker-compose.yml
├── Dockerfile
├── .gitignore
├── .env.example                # Konfigurationsdatei für Umgebungsvariablen
├── requirements.txt
└── README.md
```

## Deployment

Ein Deployment der Clustering-API kann auf verschiedene Arten erfolgen. In diesem Abschnitt werden die Verfahren für die Verwendung von Docker, Docker Compose und dem bereitgestellten `deploy.sh`-Skript beschrieben.

### Deployment mit Docker

Docker ermöglicht es Ihnen, Ihre Anwendung in einem isolierten Container auszuführen. Um Ihre Anwendung mit Docker zu deployen, führen Sie die folgenden Schritte aus:

1. Erstellen Sie das Docker-Image:

   ```bash
   docker build -t clustering-api .
2. Starten Sie den Container:

    ```bash
    docker run -p 8080:8080 clustering-api
    ```


### Deployment mit Docker Compose
Docker Compose ermöglicht die Definition und den Betrieb von Multi-Container Docker-Anwendungen. Um die Clustering-API mit Docker Compose zu deployen, gehen Sie wie folgt vor:

```bash
# Erstellen Sie die Docker-Images und starten Sie die Container
docker-compose up --build
```


### Deployment mit deploy.sh
Das deploy.sh-Skript ist ein hilfreiches Werkzeug, das die Einrichtung und das Deployment der Clustering-API automatisiert. Um dieses Skript zu verwenden:

```bash
# Stellen Sie sicher, dass das Skript ausführbar ist
chmod +x deploy.sh

# Führen Sie das Skript aus:
./deploy.sh
```

Das Skript wird automatisch LazyDocker installieren (wenn es noch nicht installiert ist), alle bestehenden Container stoppen, neue Images erstellen, die Container starten und anschließend LazyDocker für die Containerüberwachung ausführen.

## API-Dokumentation
Die RESTful Webservice-API wird über [Swagger](https://swagger.io/) dokumentiert. Die Dokumentation kann auf der folgenden
URL aufgerufen werden: http://localhost:8080/docs
