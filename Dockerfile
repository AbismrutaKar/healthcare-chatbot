# Use official Python image
FROM python:3.9

# Create a user to avoid running as root
RUN useradd -m -u 1000 user
USER user

# Add local pip binaries to PATH
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all project files into container
COPY --chown=user . /app

# Command to run Streamlit on the correct port
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
