FROM python:3.6

RUN apt-get update && apt-get install software-properties-common -y \
    && apt-get install -y zbar-tools libsm6 libxext6 libxrender1 locales \
    && locale-gen en_US.UTF-8
WORKDIR /api
COPY ./requirements.txt /api
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

COPY ./app /api/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--log-level", "warning"]

#  uvicorn app.main:app --host 0.0.0.0 --log-level warning