FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Eigen and OpenMP
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    libeigen3-dev \
    libomp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Build the C++ extension
RUN python setup.py build_ext --inplace

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
ENTRYPOINT ["streamlit", "run", "Pricer.py", "--server.port=8501", "--server.address=0.0.0.0"]
