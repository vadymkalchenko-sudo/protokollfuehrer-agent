# Use an official Python runtime as a parent image
FROM python:3.9-buster

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Specify the command to run on container startup
CMD ["python", "protokoll_bridge.py"]