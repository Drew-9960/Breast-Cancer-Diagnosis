# Use an official Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API & Streamlit ports
EXPOSE 8000 8501

# Run both API & Streamlit using a script
CMD ["sh", "start_services.sh"]
