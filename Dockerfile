FROM python:3.7
RUN pip3 install boto3
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install tensorflow
RUN mkdir /src
COPY . /src