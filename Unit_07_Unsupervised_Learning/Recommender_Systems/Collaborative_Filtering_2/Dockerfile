 # Copyright (c) Mike Harmon.
# Distributed under the terms of the Modified BSD License.

FROM jupyter/base-notebook

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

USER root

# Install all OS dependencies for fully functional notebook server
RUN apt-get update && apt-get install -yq --no-install-recommends \
    build-essential \
    git \
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    netcat \
    texlive-fonts-extra \
    texlive-fonts-recommended \
    texlive-generic-recommended \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-xetex \
    unzip \
    vim \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*


USER $NB_UID

RUN mkdir /home/$NB_USER/recommender/  \
          /home/$NB_USER/recommender/images/ && \
    fix-permissions /home/$NB_USER/recommender/ \
                    /home/$NB_USER/recommender/images/


COPY reviews_Amazon_Instant_Video_5.json.gz    /home/$NB_USER/recommender/
COPY Recommender.ipynb                         /home/$NB_USER/recommender/
COPY images/table.jpg                          /home/$NB_USER/recommender/images/table.jpg
COPY requirements.txt                          /home/$NB_USER/recommender/
COPY readFunctions.py                          /home/$NB_USER/recommender/



RUN pip install -r /home/$NB_USER/recommender/requirements.txt

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
    fix-permissions /home/$NB_USER

USER $NB_UID

