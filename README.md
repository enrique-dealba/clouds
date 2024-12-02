# Measuring Sky Transparency and Cloud Detection

### Building the Docker Image

```bash
docker build -t cloudynight:latest .
```

### Checking GPU etc

```bash
docker run --gpus all cloudynight:latest diagnostics
```

### Training Model

```bash
docker run --gpus all cloudynight:latest train \
    --config config/train_config.yaml
```

### Streamlit Frontend

```bash
docker run -p 8888:8888 cloudynight:latest streamlit
```
