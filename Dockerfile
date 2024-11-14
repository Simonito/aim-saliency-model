FROM python:3.10

RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "models/aim/main.py"]