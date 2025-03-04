# Use Python 3.9 as base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements file FIRST (to leverage Docker caching)
COPY ../requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the application code
COPY .. .

# Expose necessary ports
EXPOSE 8000 8501 5001

# Default command (can be overridden)
CMD ["bash", "deployment/start_services.sh"]
