# Set python version
ARG BASE_CONTAINER=python:3.8

# Set the base image 
FROM --platform=linux/amd64 $BASE_CONTAINER

# Sets the user name to use when running the image.
USER root
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Make a directory for app
RUN pip install --upgrade pip && pip install pyTelegramBotAPI
# RUN pip install --no-cache-dir --user -r /req.txt

# Copy source code
COPY ./telestart.py ./

# Run the application
CMD ["python", "telestart.py"]
