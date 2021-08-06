FROM tf-openslide:1.0.0

ADD ./requirements.txt /tmp/requirements.txt

ARG DEBIAN_FRONTEND=noninteractive


RUN pip install  --trusted-host ftp.daumkakao.com \
    --index-url http://ftp.daumkakao.com/pypi/simple -r /tmp/requirements.txt --no-cache-dir
