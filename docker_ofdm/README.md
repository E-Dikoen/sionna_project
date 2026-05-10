# OFDM 模块 Docker 开发环境

## 构建镜像
```bash
cd docker_ofdm
docker-compose build
```

## 启动容器
```bash
docker-compose up -d
docker exec -it sionna-ofdm bash
```

## 验证环境
```bash
python -c "import sionna; print('Sionna:', sionna.__version__)"
```

## 停止容器
```bash
docker-compose down
```
