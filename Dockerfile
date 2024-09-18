FROM python:3.9.20
ARG argument
ENV argument=$argument
WORKDIR /app
COPY requirements.txt .
#RUN apt add-apt-repository main
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN --mount=type=cache,target=/root/.cache/pip \
        pip install --default-timeout=100 -r requirements.txt
RUN pip install --upgrade huggingface_hub
RUN apt-get install -y zip unzip
COPY . .
# take env from docker container then run it with run.sh file with argument from env
CMD bash run.sh $argument
