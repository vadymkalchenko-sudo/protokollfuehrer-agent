# Use a full base image to ensure all build tools are available
FROM python:3.9-buster

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file first to leverage Docker cache
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the default command to run when the container starts
CMD ["python", "indexer.py"]