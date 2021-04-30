#!/bin/sh

set -e    # if shell fails, quit with err code 0

python3 manage.py collectstatic --noinput

uwsgi --socket :8000 --master --enable-threads --module saas.wsgi


