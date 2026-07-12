FROM python:3.11-slim

WORKDIR /app

# Install the heaviest package first, from PyTorch's CPU-only wheel index —
# this is a much smaller download than the default GPU-enabled wheel from
# PyPI, and we don't need GPU support inside the container anyway.
RUN pip install --no-cache-dir --default-timeout=180 --retries 5 \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=180 --retries 5 -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]