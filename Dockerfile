# Use a more stable Python runtime as a parent image that includes build tools
FROM python:3.9-buster

# Set the working directory in the container
WORKDIR /app

# Install PostgreSQL client development files needed for psycopg2
RUN apt-get update && apt-get install -y libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY . .

# Run indexer.py when the container launches
CMD ["python", "indexer.py"]