FROM python:3.7

RUN apt-get update
RUN apt-get install libgl1 -y

COPY ./mobilenet_ssd /app/mobilenet_ssd
COPY ./mylib /app/mylib
COPY ./videos /app/videos

COPY ./run.py /app/run.py
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install cmake==3.22.5

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 1883

CMD ["python", "run.py", "--prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel"]
