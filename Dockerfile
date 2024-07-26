# Backend Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY backend/requirements.txt /app/
RUN pip install -r requirements.txt

COPY backend/ /app/

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# Frontend Dockerfile
FROM node:14

WORKDIR /app

COPY frontend/package.json /app/
RUN npm install

COPY frontend/ /app/

CMD ["npm", "start"]
