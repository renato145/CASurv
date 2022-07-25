FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# There is no docker image for pytorch 1.9.1 atm
RUN conda install -y -c pytorch pytorch==1.9.1

RUN conda install -y -c conda-forge mamba
RUN mamba install -y -c fastchan fastai==2.5.2 jupyter paramiko
RUN mamba install -y -c conda-forge hydra-core psycopg2 optuna optuna-dashboard

# Install pysurvival
RUN apt update && apt install -y gcc-8 g++-8
ENV CXX="/usr/bin/g++-8"
ENV CC="/usr/bin/gcc-8"
RUN pip install pysurvival

# Requirements for fastai.medical.imaging
RUN mamba install -y h5py pyarrow
RUN pip install pydicom kornia opencv-python scikit-image

COPY . /workspace/casurv                                                                      
WORKDIR /workspace

RUN pip install --no-deps -e ./casurv

RUN echo "#!/bin/bash\njupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password=''" >> run_jupyter.sh
RUN chmod u+x run_jupyter.sh
CMD [ "./run_jupyter.sh" ]

