FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

WORKDIR /home/image_segmentation


# Install python3 and tools
RUN apt update && apt install -y \
    git build-essential \
    python3-dev python3-pip python3-setuptools 

#REVIEW: you probably don"t need git inside the container

RUN pip3 -q install pip --upgrade

# First copy requirements, install dependencies and then add the rest (for cache)
COPY requirements.txt ./requirements.txt

RUN pip3 install -r ./requirements.txt

COPY . ./

# Launch main script
CMD ["./main.py", "./config.yaml"]

ENTRYPOINT ["python3"]