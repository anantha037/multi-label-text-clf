FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/

# Copy models and data (these will contain the trained model and label encoder)
COPY models/ /app/models/
COPY data/processed/ /app/data/processed/

# Set environment variables
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "src.serve.main:app", "--host", "0.0.0.0", "--port", "8000"]
