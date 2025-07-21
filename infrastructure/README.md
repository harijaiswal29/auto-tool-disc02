# Infrastructure Directory

This directory contains infrastructure and deployment configuration files for the Auto Tool Discovery project.

## Contents

- `docker-compose.yml` - Docker Compose configuration for running PostgreSQL and other services

## Usage

To start the PostgreSQL service:
```bash
cd infrastructure
docker-compose up -d
```

To stop the services:
```bash
docker-compose down
```