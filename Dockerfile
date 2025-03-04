# Use Python 3.9 as the base image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app/

# Set the default command to start the API server
CMD ["uvicorn", "src.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]

