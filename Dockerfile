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
RUN apt-get -f -y install default-jdk

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
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs

# Examples and data
#RUN mkdir GWU_data_mining && \
#    cd GWU_data_mining && \
#    git init && \
#    git remote add origin https://github.com/jphall663/GWU_data_mining.git && \
#    git pull origin master && \
#    git lfs install && \
#    git lfs track '*.jpg' '*.png' '*.csv' '*.sas7bdat'

# XGBoost
RUN apt-get -y install gcc g++ make && \
    conda install libgcc && \
    git clone --recursive https://github.com/dmlc/xgboost.git && \
    cd xgboost && \
    make && \
    cd python-package && \
    python setup.py install --user
        
# GraphViz
RUN apt-get -y install graphviz

###############################
RUN pip install matplotlib==2.0.2

# Launchbot labels
LABEL name.launchbot.io="ormlanders/interpretable-ml-python-xgboost-h2o"
LABEL workdir.launchbot.io="/usr/workdir"
LABEL 8888.port.launchbot.io="Jupyter Notebook"

# Set the working directory
WORKDIR /usr/workdir

# Add in notebook for testing
COPY xgboost_pdp_ice.ipynb /usr/workdir/xgboost_pdp_ice.ipynb
COPY default_of_credit_card_clients.xls /usr/workdir/default_of_credit_card_clients.xls

# Expose the notebook port
EXPOSE 8888

# Start the notebook server
CMD jupyter notebook --no-browser --port 8888 --ip=* --NotebookApp.token='' --NotebookApp.disable_check_xsrf=True
