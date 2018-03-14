# Install and start docker
# Then this file may be used to create a Docker container using:
# $ docker build anaconda_py35_h2o_xgboost_graphviz
# $ docker run -i -t -p 8888:8888 <image_id> /bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet && /opt/conda/bin/jupyter notebook --notebook-dir=/mli-resources --ip='*' --port=8888 --no-browser"
# $ docker cp path/to/train.csv <container_id>:/mli-resources/data
# Open a browser and navigate to localhost:8888

# Base debian system
FROM debian:8.5
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Update OS
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

# Anaconda Python 3.5
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

# Java
RUN apt-get -y -f install default-jdk

# H2o deps
RUN pip install requests && \
    pip install tabulate && \
    pip install six && \
    pip install future && \
    pip install colorama

# H2o
RUN pip uninstall h2o || true && \
    pip install -f https://h2o-release.s3.amazonaws.com/h2o/rel-weierstrass/2/Python/h2o-3.14.0.2-py2.py3-none-any.whl --trusted-host h2o-release.s3.amazonaws.com h2o

# Git
RUN apt-get -y install git

# Examples
RUN git clone https://github.com/h2oai/mli-resources.git

# XGBoost
RUN apt-get -y install gcc g++ make && \
    conda install -y libgcc && \
    git clone --recursive https://github.com/dmlc/xgboost.git && \
    cd xgboost && \
    make && \
    cd python-package && \
    python setup.py install --user

# GraphViz
RUN apt-get -y install graphviz
