# Sky Transparency and Cloud Detection

### Building the Docker Image

```bash
docker build -t cloudynight:latest .
```

### Checking GPU

```bash
docker run --gpus all cloudynight:latest diagnostics
```

### Streamlit Frontend

1. Cloud Detection
```bash
docker run -p 8888:8888 cloudynight:latest detection
```

2. Masking
```bash
docker run -p 8888:8888 cloudynight:latest streamlit
```

### Running PyTests

```bash
docker run cloudynight:latest tests
```
