# Install and start docker
# Then this file may be used to create a Docker container using:
# $ sudo docker build -t iml anaconda_py36_h2o_xgboost_graphviz_shap
# $ sudo docker run -i -t -p 8888:8888 iml:latest /bin/bash -c "/opt/conda/bin/jupyter notebook --notebook-dir=/interpretable_machine_learning_with_python --allow-root --ip='*' --port=8888 --no-browser"
# Open a browser and navigate to localhost:8888

# Base debian system
FROM debian:9.0
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Update OS
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

# Anaconda Python 3.6
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh
ENV PATH /opt/conda/bin:$PATH

# Java
RUN apt-get -y install default-jre

# H2o deps
RUN pip install requests tabulate six future colorama

# H2o
RUN pip install h2o==3.26.0.3

# Pandas
RUN pip install pandas==0.23.4

# XGBoost
RUN apt-get update --fix-missing && \
    apt-get -y install gcc g++ make && \
    conda install -y libgcc && \
    pip install xgboost==0.7.post3

# Shap
RUN pip install shap==0.28.0

# Seaborn
RUN pip install matplotlib==2.1.0 seaborn==0.8.1

# Git
RUN apt-get -y install git

# GraphViz
RUN apt-get -y install graphviz

# Examples
RUN git clone https://github.com/jphall663/interpretable_machine_learning_with_python.git
