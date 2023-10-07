# Clustering-API

Das Projekt stellt eine API bereit, mit der Datenpunkte mithilfe des Clustering-Algorithmus gruppiert werden können. Benutzer können ihre Daten in Form von Excel-Dateien hochladen, und die API gibt die geclusterten Datenpunkte zurück.

## Inhaltsverzeichnis

- [Voraussetzungen](#Voraussetzungen)
- [Installation und Einrichtung](#installation-und-einrichtung)
- [Repository-Struktur](#repository-struktur)
- [Deployment](#deployment)
- [API-Dokumentation](#api-dokumentation)
- [Request](#request)
- [Response](#response)

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
├── app/                          # Hauptanwendungsverzeichnis
│ ├── routers/                    # FastAPI-Endpunkte
│ │
│ ├── models/                     # Datenmodelle und -schemata
│ │
│ ├── services/                   # Dienstprogramme und Services
│ │   
│ └── main.py                     # Hauptanwendungsdatei
│
├── temp_files/                   # Verzeichnis für hochgeladene Dateien 
│
├── tests/                        # Testverzeichnis
│
├── .github/                      # GitHub-spezifische Dateien
│ └── workflows/                  # CI/CD-Workflows
│
├── deploy.sh                     # Automatisierte Bereitstellung des Containers und Lazydocker 
├── docker-compose.yml
├── Dockerfile
├── .gitignore
├── .env.example                  # Konfigurationsdatei für Umgebungsvariablen
├── requirements.txt
└── README.md
```

## Deployment

Ein Deployment der Clustering-API kann auf verschiedene Arten erfolgen. In diesem Abschnitt werden die Verfahren für die Verwendung von Docker, Docker Compose und dem bereitgestellten `deploy.sh`-Skript beschrieben.

### Deployment mit Docker

Docker ermöglicht es Ihnen, Ihre Anwendung in einem isolierten Container auszuführen. Um Ihre Anwendung mit Docker zu deployen, führen Sie die folgenden Schritte aus:

1. Erstellen Sie das Docker-Image:

   ```bash
   docker build -t clustering-api --build-arg DEV_MODE=True .
2. Starten Sie den Container:

    ```bash
    docker run -p 8080:8080 --env-file .env clustering-api
    ```


### Deployment mit Docker Compose
Docker Compose ermöglicht die Definition und den Betrieb von Multi-Container Docker-Anwendungen. Um die Clustering-API mit Docker Compose zu deployen, gehen Sie wie folgt vor:

```bash
# Erstellen Sie die Docker-Images und starten Sie die Container
docker-compose --env-file .env up --build
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

## Request

Um den Endpunkt zu nutzen, sendet man eine POST-Anfrage mit folgenden Parametern:

- **file**: Die Excel- oder CSV-Datei zum Clustern (Pflicht). Als Formdaten senden.

- **kCluster** (optional): Die gewünschte Anzahl an Clustern. Wenn nicht angegeben, wird die optimale Zahl automatisch bestimmt.

- **column1** (optional): Auswahl der ersten Spalte. Wenn nicht angegeben 0 (0-Indexierung)

- **column2** (optional): Auswahl der zweiten Spalte. Wenn nicht angegeben 1 (0-Indexierung)

- **distance_metric** (optional): Die gewünschte Distanzmetrik. Standardmäßig die euklidische Distanz (EUCLIDEAN, MANHATTAN, JACCARDS).

- **clusterDetermination** (optinal): Die gewünschte Methode zur Bestimmung der optimalen Cluster-Anzahl (ELBOW, SILHOUETTE).

### Möglicher Aufruf über TypeScript:
```typescript
import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class ClusteringService {
  private apiUrl = 'http://localhost:8080/clustering/perform-kmeans-clustering/';

  constructor(private http: HttpClient) { }

  performClustering(
    file: File,
    column1?: string | number,
    column2?: string | number,
    kCluster?: number,
    distanceMetric: string = 'EUCLIDEAN',
    clusterDetermination: string = 'ELBOW'
  ) {
    const formData: FormData = new FormData();
    formData.append('file', file);
    let params = new HttpParams()
      .set('distanceMetric', distanceMetric)
      .set('clusterDetermination', clusterDetermination);

    if (column1 !== undefined) {
      params = params.set('column1', column1.toString());
    }
    if (column2 !== undefined) {
      params = params.set('column2', column2.toString());
    }
    if (kCluster !== undefined) {
      params = params.set('kCluster', kCluster.toString());
    }

    return this.http.post(this.apiUrl, formData, { params: params });
  }
}
```

## Response

Die API antwortet mit einem JSON-Objekt, das Folgendes enthält:

- **user_id**: Aktuell nur Platzhalter
- **request_id**: Aktuell nur Platzhalter
- **name**: Name der Ausgabe
- **cluster**: Eine Liste mit Clustern, die wiederum eine Liste mit den zugehörigen Datenpunkten enthalten
- **x_label**: Name der X-Achse
- **y_label**: Name der Y-Achse
- **iterations**: Anzahl der Iterationen, die für das Clustering benötigt wurden
- **used_distance_metric**: Die verwendete Distanzmetrik
- **used_optK_method**: Die verwendete Methode zur Bestimmung der optimalen Anzahl an Clustern
- **clusters_elbow**: Die Anzahl an Clustern, die durch die Elbow-Methode bestimmt wurde
- **clusters_silhouette**: Die Anzahl an Clustern, die durch die Silhouette-Methode bestimmt wurde

```json  
{
  "user_id": 0,
  "request_id": 0,
  "name": "string",
  "cluster": [
    {
      "clusterNr": 0,
      "centroid": {
        "x": 0,
        "y": 0
      },
      "points": [
        {
          "additionalProp1": 0,
          "additionalProp2": 0,
          "additionalProp3": 0
        }
      ]
    }
  ],
  "x_label": "string",
  "y_label": "string",
  "iterations": 0,
  "used_distance_metric": "string",
  "used_optK_method": "string",
  "clusters_elbow": 0,
  "clusters_silhouette": 0
}
