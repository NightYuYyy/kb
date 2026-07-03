FROM python:3.12-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY kb_core.py kb_cli.py kb_web.py ./
COPY templates/ templates/

# 创建数据和配置目录
RUN mkdir -p /app/data

# Web 服务端口
EXPOSE 8765

# 默认启动 Web 服务（docker-compose 中可覆盖为 CLI 模式）
CMD ["python", "kb_web.py"]
