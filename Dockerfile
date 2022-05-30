###########
# BUILDER #
###########

# pull official base image
FROM python:3.8-slim as builder
LABEL stage=builder

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN mkdir -p /home/app
RUN adduser --system app --group
ENV APP_HOME=/home/app/
WORKDIR $APP_HOME
COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY . $APP_HOME

RUN chown -R app:app $APP_HOME

# change to the app user
USER app

CMD ["uvicorn", "online_inference.main:app", "--reload", "--host", "0.0.0.0"]

