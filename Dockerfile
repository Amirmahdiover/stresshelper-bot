FROM python:3.8-slim

# Install system dependencies
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    python3-venv \
    ffmpeg \
    git; \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install essential tools
RUN pip3 install --no-cache-dir -U pip wheel setuptools==59.5.0

# Copy requirements and install Python dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -i https://pypi.org/simple -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

# Copy the application code into the container
COPY . /master-ai-bot

# Set the working directory inside the container to where your bot directory is
WORKDIR /master-ai-bot

# Set the Python path so Python knows where to find the bot package
ENV PYTHONPATH=/master-ai-bot

# Default command to run your bot (adjust path if needed)
CMD ["python3", "bot/bot.py"]
