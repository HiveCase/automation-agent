FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV AIPROXY_TOKEN=eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjEwMDMxMzJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.JxGJii5rnsM3Nr-Vnx3RnesfDklwNJZBXPypaN6HqyE

EXPOSE 8000

CMD ["python", "app.py"]
