FROM python:3.9.7

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


WORKDIR /code
RUN mkdir -p /code/gulf_score

COPY . /code

RUN pip install --upgrade pip
RUN pip install --default-timeout=3000 -r /code/requirements.txt


CMD [ "python3", "start.py" ]