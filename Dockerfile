FROM python:3.11-slim-bullseye


# Install system dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3-pip \
        build-essential \
        python3-venv \
        ffmpeg \
        git \
    || apt-get install -f -y; \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install other essential tools
RUN pip3 install -U pip wheel setuptools==59.5.0

# Copy requirements and install Python dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt && rm -rf /tmp/requirements.txt

# Copy the application code
COPY . /code

# Set the working directory inside the container
WORKDIR /code

# Default command to run your bot (adjust if necessary)
CMD ["python3", "bot/bot.py"]
