FROM python:3.7

# Install dependencies.
ADD requirements.txt .
RUN pip install -r requirements.txt

# Copy local code to the container image.
WORKDIR /app
COPY . .

ENV PORT 8080

CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 0 main:app
