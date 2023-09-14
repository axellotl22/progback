## Programmierprojekt Backend: K-means & Decision Trees
Dieses Repository enthält eine RESTful API, die mit Python und FastAPI entwickelt wurde. Die API ermöglicht es Benutzern, Daten für K-means Clustering und Entscheidungsbäume hochzuladen und zu verarbeiten. Es verwendet eine Github Flow Branching-Strategie und Github Actions, um den Code mit pylint zu linten und Unit-Tests mit pytest auszuführen.

## Repository-Struktur
````bash
Progback/
│
├── app/                            # Hauptanwendungsverzeichnis
│   ├── api/                        # API-spezifische Module
│   │   ├── endpoints/              # FastAPI-Endpunkte
│   │   │   ├── clustering.py       # Endpunkt für das Hochladen und Clustern von Dateien
│   │   │   └── decision_tree.py    # Endpunkt für das Hochladen und Erstellen von Entscheidungsbäumen
│   │   └── main.py                 # Haupt-API-Datei
│   │
│   ├── core/                       # Kernkonfigurationsdateien und -dienstprogramme
│   │   ├── config.py               # Konfigurationsdatei
│   │   └── logger.py               # Logger-Konfiguration
│   │
│   ├── services/                   # Dienstprogramme und Services
│   │   ├── clustering.py           # Dienstprogramme für KMeans-Clustering
│   │   ├── decision_tree.py        # Dienstprogramme für Entscheidungsbäume
│   │   └── file_service.py         # Dienstprogramme zum Einlesen von Excel/CSV-Dateien
│   │
│   └── main.py                     # Hauptanwendungsdatei
│
├── tests/                          # Testverzeichnis
│   ├── __init__.py
│   ├── test_app.py                 # Haupttestdatei
│
├── .github/                        # GitHub-spezifische Dateien
│   └── workflows/                  # CI/CD-Workflows
│
├── docker-compose.yml         
├── Dockerfile                 
├── .gitignore                 
├── .env.example                    # Konfigureationsdatei für Umgebungsvariablen                        
├── requirements.txt           
└── README.md                                  
``````

## Lokale API-Einrichtung
Um die API lokal auszuführen, benötigen Sie Python 3.10. Installieren Sie dies zuerst. Danach:

````bash
Copy code
python3 -m venv progback # Virtuelle Umgebung erstellen
source progback/bin/activate # Virtuelle Umgebung aktivieren
pip3 install -r requirements.txt # Python-Pakete installieren

uvicorn app.main:app --reload # API starten
``````
Jetzt können Sie auf _http://127.0.0.1:8080/docs zugreifen.

## Lokale Docker-Ausführung
Um Docker-Container zu erstellen und auszuführen, muss Docker installiert sein. Danach:

````bash
Copy code
docker build -t progback . # Docker-Container erstellen
docker run -it -p 8080:8080 progback # Docker-Container ausführen
``````
Jetzt können Sie auf _http://127.0.0.1:8080/docs zugreifen.