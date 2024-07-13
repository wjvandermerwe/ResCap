# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /usr/src/app

# Install PyTorch
RUN pip install torch==1.8.0 torchvision==0.10.0

# Install auton_survival
RUN pip install auton-survival

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Run a command to verify the installation
CMD ["python", "-c", "import torch; print('PyTorch version:', torch.__version__)"]