#!/bin/bash

# Überprüfe, ob LazyDocker installiert ist. Wenn nicht, installiere es.
if ! command -v lazydocker &> /dev/null; then
    echo "LazyDocker ist nicht installiert. Installation läuft..."
    curl https://raw.githubusercontent.com/jesseduffield/lazydocker/master/scripts/install_update_linux.sh | bash
else
    echo "LazyDocker ist bereits installiert."
fi

# Stoppe existierenden Container
docker-compose down

# Baue neue Container Images
docker-compose build

# Starte Container im Hintergrund 
docker-compose up --env-file .env -d

# Entferne nicht mehr genutzte Container
docker image prune -f

# Entferne nicht mehr genutzte Volumes 
docker volume prune -f

# Starte LazyDocker für Container Überwachung
lazydocker
