FROM nvidia/cuda:11.8.0-base-ubuntu22.04
RUN apt update -y

# Upgrade fails with ubuntu 22.04 at the time of writing
# RUN apt upgrade -y

RUN apt install wget -y
RUN apt install git -y

WORKDIR /usr/src
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda.sh
RUN bash ./miniconda.sh -b -p $HOME/miniconda
RUN $HOME/miniconda/bin/conda init

RUN ["/bin/bash", "-c", "source ~/.bashrc"]
ENV PATH /root/miniconda/bin:$PATH

RUN conda update --all

# Following command takes a while
RUN conda create -y -n pytorch3d python=3.9 
RUN echo "conda activate pytorch3d" >> ~/.bashrc

SHELL ["conda", "run", "-n", "pytorch3d", "/bin/bash", "-c"]

RUN conda install --name pytorch3d pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
RUN conda install --name pytorch3d -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install --name pytorch3d pytorch3d -c pytorch3d

RUN pip install --upgrade pip
RUN pip install scikit-image matplotlib imageio black isort flake8 flake8-bugbear flake8-comprehensions ipywidgets PyYAML trimesh

RUN git clone https://github.com/openai/shap-e.git
WORKDIR /usr/src/shap-e

# pip install -e . installs shap_e into the default conda environment..
RUN ["conda", "run", "-n", "pytorch3d", "pip", "install", "-e", "."]

# store the generated .glb here
RUN mkdir shap_e/output

# docker build -t shap-e-full .
# docker run -v D:\AI\shap-e\shap-e:/shap-e --gpus=all  -dit --name shap-e-full shap-e-full
# docker exec -it shap-e-full "/bin/bash"


#docker run -v D:\AI\shap-e\shap-e:/shap-e-src --gpus=all  -dit --name shap-e-full shap-e-full