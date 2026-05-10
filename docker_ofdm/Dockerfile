# OFDM 模块开发环境
FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git vim wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install sionna==1.2.2 \
    matplotlib scipy numpy jupyter \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN python -c "import sionna; print('Sionna:', sionna.__version__)"

CMD ["/bin/bash"]
