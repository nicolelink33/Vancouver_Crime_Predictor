FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

# Copy the conda lock file for reproducible environment
COPY conda-lock.yml /tmp/conda-lock.yml

# Install conda-lock, then use it to install all dependencies
# The lock file contains pinned versions for reproducibility
RUN conda install -n base -c conda-forge conda-lock=2.5.8 --yes && \
    conda-lock install --name base /tmp/conda-lock.yml && \
    conda clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
