FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pytest pytest-cov

# Copy project
COPY cloudynight/ /app/cloudynight/
COPY scripts/ /app/scripts/
COPY frontend/ /app/frontend/
COPY tests/ /app/tests/
COPY cloudynight/models/kde_models.pkl /app/cloudynight/models/

# Make start.sh executable
RUN chmod +x /app/scripts/start.sh

# Expose Streamlit port
EXPOSE 8888

# Define entrypoint
ENTRYPOINT ["/app/scripts/start.sh"]
