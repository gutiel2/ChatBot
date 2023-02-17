# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9-slim

ARG TENSORFLOW_VERSION=v2.9.1 

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install --upgrade pip
#RUN python -m pip install tensorflow-macos
#RUN python -m pip install tensorflow-metal
#RUN pip install --verbose https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-2.11.0-cp39-cp39-macosx_10_14_x86_64.whl
RUN python -m pip install -r requirements.txt

# Create a working directory and copy the files to app
WORKDIR /app
COPY . /app

CMD [ "python3", "manage.py", "runserver", "0.0.0.0:8000"]
