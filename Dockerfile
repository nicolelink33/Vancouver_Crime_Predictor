FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

# Copy the conda lock file for reproducible environment
COPY conda-lock.yml /tmp/conda-lock.yml

# Install dependencies using conda-lock
# Using mamba for faster installation and micromamba-based approach
RUN mamba install -n base -c conda-forge conda-lock --yes && \
    conda-lock install --name base --no-validate-platform /tmp/conda-lock.yml && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
