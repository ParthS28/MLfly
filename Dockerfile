FROM python:3.8-slim-buster

ENV PATH="/scripts:${PATH}"

COPY ./requirements.txt /requirements.txt
# RUN apk add --update --no-cache --virtual .tmp gcc libc-dev linux-headers
# RUN apt-get install python3-dev build-base linux-headers pcre-dev
RUN apt-get update && apt-get install -y git python3-dev gcc wget \
    && rm -rf /var/lib/apt/lists/*
# RUN apt-get update && apt-get install -y uwsgi
RUN pip install -r requirements.txt
# RUN apk del .tmp 

RUN mkdir /app
COPY ./saas /app
WORKDIR /app

COPY ./scripts /scripts
RUN chmod +x /scripts/*

RUN mkdir -p /vol/web/media
RUN mkdir -p /vol/web/static

# RUN adduser -D user
# RUN chown -R user:user /vol
# RUN chmod -R 755 /vol/web
# USER user

CMD ["entrypoint.sh"]