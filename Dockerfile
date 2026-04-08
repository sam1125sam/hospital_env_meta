FROM python:3.11-slim

LABEL maintainer="hospital-er-env"
LABEL description="Hospital ER Triage & Resource Allocation Environment"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Set PYTHONPATH so imports work from any subdirectory
ENV PYTHONPATH=/app

# Default: run all baseline agents
CMD ["python", "baseline.py"]
