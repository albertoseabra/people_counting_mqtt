FROM python:3.7

RUN apt-get update
RUN apt-get install libgl1 -y

RUN pip install cmake==3.22.5

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

COPY ./mobilenet_ssd /app/mobilenet_ssd
COPY ./mylib /app/mylib
COPY ./videos /app/videos

COPY ./run.py /app/run.py

WORKDIR /app

EXPOSE 1883

CMD ["python", "run.py"]
