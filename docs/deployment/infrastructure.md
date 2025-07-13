# Infrastructure & Deployment

## Container Specifications

### Docker Compose Configuration
```yaml
version: '3.8'

services:
  api:
    image: auto-tool-disc/api:latest
    replicas: 3
    resources:
      limits:
        cpus: '2'
        memory: 2G
      reservations:
        cpus: '1'
        memory: 1G
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - LOG_LEVEL=info
    ports:
      - "8000:8000"
  
  worker:
    image: auto-tool-disc/worker:latest
    replicas: 5
    resources:
      limits:
        cpus: '4'
        memory: 4G
    environment:
      - QUEUE_URL=${QUEUE_URL}
      - MCP_TIMEOUT=30
  
  mcp-gateway:
    image: auto-tool-disc/mcp-gateway:latest
    replicas: 2
    ports:
      - "9000:9000"
```

## Kubernetes Deployment

### API Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: auto-tool-disc-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: auto-tool-disc-api
  template:
    metadata:
      labels:
        app: auto-tool-disc-api
    spec:
      containers:
      - name: api
        image: auto-tool-disc/api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: auto-tool-disc-api
spec:
  selector:
    app: auto-tool-disc-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: auto-tool-disc-config
data:
  LOG_LEVEL: "info"
  CACHE_TTL: "300"
  MAX_WORKERS: "10"
  # Retry Configuration
  RETRY_MAX_ATTEMPTS: "5"
  RETRY_BASE_DELAY: "1.0"
  RETRY_MAX_DELAY: "16.0"
  RETRY_JITTER_FACTOR: "0.2"
  # Circuit Breaker
  CIRCUIT_BREAKER_FAILURE_THRESHOLD: "5"
  CIRCUIT_BREAKER_RECOVERY_TIMEOUT: "30"
  CIRCUIT_BREAKER_HALF_OPEN_REQUESTS: "3"
  # Connection Pool
  CONNECTION_POOL_MAX_SIZE: "10"
  CONNECTION_POOL_IDLE_TIMEOUT: "300"
  CONNECTION_POOL_HEALTH_CHECK_INTERVAL: "60"
```

## Environment Configuration

### Production Environment
```bash
# Database
DATABASE_URL=postgresql://user:pass@db:5432/autotooldisc
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://redis:6379/0
REDIS_MAX_CONNECTIONS=50

# MCP Registry
MCP_REGISTRY_URL=http://mcp-registry:8080
MCP_TIMEOUT=30

# Security
JWT_SECRET=${JWT_SECRET}
API_KEY_SALT=${API_KEY_SALT}

# Monitoring
METRICS_ENABLED=true
TRACING_ENABLED=true
LOG_LEVEL=info

# Performance
CACHE_TTL=300
CONNECTION_POOL_SIZE=20
WORKER_CONCURRENCY=10

# Retry Configuration
RETRY_POLICY_TYPE=exponential_backoff
RETRY_MAX_ATTEMPTS=5
RETRY_BASE_DELAY=1.0
RETRY_MAX_DELAY=16.0
RETRY_JITTER_FACTOR=0.2

# Circuit Breaker
CIRCUIT_BREAKER_ENABLED=true
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=30
CIRCUIT_BREAKER_HALF_OPEN_REQUESTS=3

# Connection Pool
CONNECTION_POOL_MAX_CONNECTIONS=10
CONNECTION_POOL_IDLE_TIMEOUT=300
CONNECTION_POOL_HEALTH_CHECK_INTERVAL=60

# Retry Metrics
RETRY_METRICS_ENABLED=true
RETRY_METRICS_WINDOW_SIZE=1000
RETRY_ALERT_SUCCESS_RATE_THRESHOLD=0.5
```

### Development Environment
```bash
# Database
DATABASE_URL=sqlite:///./dev.db

# Redis
REDIS_URL=redis://localhost:6379/0

# MCP Registry
MCP_REGISTRY_URL=http://localhost:8080

# Security
JWT_SECRET=dev-secret

# Monitoring
METRICS_ENABLED=false
TRACING_ENABLED=false
LOG_LEVEL=debug

# Performance
CACHE_TTL=60
CONNECTION_POOL_SIZE=5
WORKER_CONCURRENCY=2

# Retry Configuration (Development)
RETRY_POLICY_TYPE=fixed_delay
RETRY_MAX_ATTEMPTS=3
RETRY_BASE_DELAY=0.5
RETRY_MAX_DELAY=2.0

# Circuit Breaker (Development)
CIRCUIT_BREAKER_ENABLED=false  # Disabled for easier debugging

# Connection Pool (Development)
CONNECTION_POOL_MAX_CONNECTIONS=5
CONNECTION_POOL_IDLE_TIMEOUT=60

# Retry Metrics (Development)
RETRY_METRICS_ENABLED=true
RETRY_METRICS_WINDOW_SIZE=100
```

## Infrastructure Requirements

### Compute Resources
**API Servers** (3 instances):
- vCPU: 2
- RAM: 4GB
- Storage: 20GB SSD

**Workers** (5 instances):
- vCPU: 4
- RAM: 8GB
- Storage: 50GB SSD

**Database** (HA pair):
- vCPU: 8
- RAM: 16GB
- Storage: 500GB SSD
- IOPS: 10,000

### Storage Requirements
- Database: 500GB SSD, 10K IOPS
- Object Storage: 1TB for logs/metrics
- Backup Storage: 2TB
- Container Registry: 100GB

### Network Architecture
- Load Balancer: Application Load Balancer
- CDN: For static assets
- VPN: Site-to-site for admin access
- Private subnets: For internal services

## CI/CD Pipeline

### GitHub Actions Workflow
```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest tests/ --cov=src
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker images
        run: |
          docker build -t auto-tool-disc/api:${{ github.sha }} .
          docker push auto-tool-disc/api:${{ github.sha }}
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/api api=auto-tool-disc/api:${{ github.sha }}
          kubectl rollout status deployment/api
```

## Deployment Checklist

### Pre-deployment
- [ ] Run all tests
- [ ] Update version numbers
- [ ] Review configuration changes
- [ ] Backup database
- [ ] Notify team

### Deployment
- [ ] Deploy to staging
- [ ] Run smoke tests
- [ ] Deploy to production (rolling update)
- [ ] Monitor metrics
- [ ] Verify health checks

### Post-deployment
- [ ] Run integration tests
- [ ] Check error rates
- [ ] Monitor performance
- [ ] Update documentation
- [ ] Close deployment ticket

## Rollback Procedures

1. **Identify Issue**
   - Monitor error rates
   - Check health endpoints
   - Review logs

2. **Initiate Rollback**
   ```bash
   kubectl rollout undo deployment/api
   kubectl rollout status deployment/api
   ```

3. **Verify Rollback**
   - Check application version
   - Run smoke tests
   - Monitor metrics

4. **Post-mortem**
   - Document issue
   - Create fix
   - Plan re-deployment

## Scaling Strategies

### Horizontal Pod Autoscaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: auto-tool-disc-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Database Scaling
- Read replicas for query load
- Connection pooling
- Query optimization
- Partitioning for large tables