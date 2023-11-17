# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:22.11-py3
WORKDIR /app

COPY requirements.txt /app
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

COPY .. /app

ENTRYPOINT ["python3"]
CMD ["main.py"]


