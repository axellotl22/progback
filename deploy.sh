#!/bin/bash

# Installiere LazyDocker für Container Überwachung
curl https://raw.githubusercontent.com/jesseduffield/lazydocker/master/scripts/install_update_linux.sh | bash

# Stoppe existierenden Container
docker-compose down

# Baue neue Container Images
docker-compose build

# Starte Container im Hintergrund 
docker-compose up -d

# Entferne nicht mehr genutzte Container
docker image prune -f

# Entferne nicht mehr genutzte Volumes 
docker volume prune -f

# Starte LazyDocker für Container Überwachung
lazydocker