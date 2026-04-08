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

EXPOSE 7860

# Hugging Face Spaces expects a long-running web server bound to 0.0.0.0:7860.
CMD ["python", "-m", "uvicorn", "dashboard_api:app", "--host", "0.0.0.0", "--port", "7860"]
