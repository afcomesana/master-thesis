#!/bin/sh

# change permissions AFTER the docker volume is mounted
chown -R appuser:appuser /app/staticfiles

# drop root privileges
exec gosu appuser "$@"