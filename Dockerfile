FROM continuumio/anaconda3

LABEL maintainer="Raphael Fischer"

RUN apt-get -y install nano zip unzip screen gcc htop texlive-latex-extra

RUN git clone https://github.com/raphischer/imagenet-energy-efficiency
RUN cd imagenet-energy-efficiency

RUN conda env create --name mlee
RUN conda activate mlee
RUN pip install numpy dash plotly reportlab PyMuPDF qrcode