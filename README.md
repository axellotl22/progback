## Programmierprojekt Backend: K-means Clustering
Dieses Repository enthält eine RESTful API, die mit Python und FastAPI entwickelt wurde. Die API ermöglicht es Benutzern, Daten für K-means Clustering hochzuladen und zu verarbeiten. Es verwendet eine Github Flow Branching-Strategie und Github Actions, um den Code mit pylint zu linten und Unit-Tests mit pytest auszuführen.

## Repository-Struktur
````bash
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
├── tests/                      # Testverzeichnis
│ ├── init.py
│ ├── test_app.py               # Haupttestdatei
│
├── .github/                    # GitHub-spezifische Dateien
│ └── workflows/                # CI/CD-Workflows
│
├── docker-compose.yml
├── Dockerfile
├── .gitignore
├── .env.example # Konfigurationsdatei für Umgebungsvariablen
├── requirements.txt
└── README.md
````
## Lokale API-Einrichtung
Um die API lokal auszuführen, benötigen Sie Python 3.10. Installieren Sie dies zuerst. Danach:

```bash
python3 -m venv progback # Virtuelle Umgebung erstellen
source progback/bin/activate # Virtuelle Umgebung aktivieren
pip3 install -r requirements.txt # Python-Pakete installieren

uvicorn app.main:app --reload # API starten
``````
Jetzt können Sie auf _http://127.0.0.1:8080/docs zugreifen.

## Lokale Docker-Ausführung
Um Docker-Container zu erstellen und auszuführen, muss Docker installiert sein. Danach:

````bash
docker-compose build # Docker-Image erstellen
docker-compose up # Docker-Container starten
``````
Jetzt können Sie auf _http://127.0.0.1:8080/docs zugreifen.