FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

# Install LaTeX packages for Quarto rendering
USER root
RUN apt-get update && apt-get install -y \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    lmodern \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER ${NB_UID}

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Install packages from environment.yml using mamba
# Using --name base to update the base environment
RUN mamba env update --name base --file /tmp/environment.yml && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
