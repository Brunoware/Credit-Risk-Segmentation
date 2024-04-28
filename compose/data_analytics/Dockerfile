FROM python:3.11.8-alpine3.19

ENV PYHONBUFFERED 1

WORKDIR /app

COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .