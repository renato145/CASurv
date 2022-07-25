FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
COPY . /workspace/posthocos
WORKDIR /workspace
RUN mv posthocos/notebooks .
RUN conda install -y -c pytorch -c fastai fastai=1.0.61 \
    && conda install -y jupyter bcolz seaborn scikit-learn \
    && conda install -y -c conda-forge nibabel ipyvolume
RUN pip install -e ./posthocos
RUN echo '#!/bin/bash\njupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser' >> run_jupyter.sh
RUN chmod u+x run_jupyter.sh
CMD [ "./run_jupyter.sh" ]
