# simple-jev Docker 镜像（CUDA 12.4）

把 [featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) 打包成可直接运行的 Docker 镜像，
由 GitHub Actions 在 Ubuntu runner 上构建并推送到镜像仓库。

这个仓库**只**包含镜像定义（`Dockerfile` + workflow），不含上游源码 —— 构建时会自动把上游仓库克隆进镜像。

## 镜像的定位

| 项目 | 说明 |
| --- | --- |
| 🎯 目标环境 | **仅 GPU（CUDA 12.4）**，基于 `nvidia/cuda:12.4.1-runtime-ubuntu22.04`，不提供 CPU 版本 |
| ✅ 包含 | Python 3.12 + 独立 venv、PyTorch（cu124 构建）、Transformers / FastAPI 全量依赖、simple-jev 源码与 `simple-jev` 命令行入口 |
| ❌ 不包含 | **任何模型权重**。不预下载、不内置、不声明模型目录 |
| 🚀 默认命令 | `tail -f /dev/null` —— 容器起来后只是空转保活，服务由你自己启动 |
| 📦 端口 | 声明 `EXPOSE 8000`，是否映射由你决定 |
| 🔌 模型挂载 | 由你手动 `-v` 挂载，镜像不做任何预设 |

## 快速开始

镜像地址形如 `ghcr.io/<你的GitHub用户名>/simple-jev-docker:latest`（用户名全小写）。

前置条件：宿主机装好 NVIDIA 驱动和 nvidia-container-toolkit，且驱动支持 CUDA 12.4
（数据中心驱动 ≥ 470，消费级 Linux 驱动一般需 ≥ 525）。用 `nvidia-smi` 确认：

```bash
nvidia-smi          # 右上角 "CUDA Version" 需 ≥ 12.4
docker run --rm --gpus all ghcr.io/<owner>/simple-jev-docker:latest nvidia-smi
```

### 1. 启动容器并保持空转

```bash
docker run -d --name simple-jev --gpus all \
  -p 8000:8000 \
  -v /宿主机/模型目录:/models \
  ghcr.io/<owner>/simple-jev-docker:latest
```

此时容器内只有 `tail -f /dev/null` 在跑，模型和服务都没启动。

### 2. 自己启动服务

**方式 A：进入容器启动（推荐）** —— 容器一直活着，服务可以随时停掉再起：

```bash
docker exec -it simple-jev simple-jev \
  --model /models/Qwen3.5-0.8B \
  --device cuda --dtype bfloat16 \
  --host 0.0.0.0 --port 8000
```

**方式 B：启动时直接替换命令** —— 不保留空转：

```bash
docker run --rm --name simple-jev --gpus all -p 8000:8000 \
  -v /宿主机/模型目录:/models \
  ghcr.io/<owner>/simple-jev-docker:latest \
  simple-jev --model /models/Qwen3.5-0.8B --device cuda --dtype bfloat16 --host 0.0.0.0 --port 8000
```

**方式 C：docker compose**，用 `command:` 覆盖：

```yaml
services:
  simple-jev:
    image: ghcr.io/<owner>/simple-jev-docker:latest
    container_name: simple-jev
    ports:
      - "8000:8000"
    volumes:
      - /宿主机/模型目录:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    # command: ["tail", "-f", "/dev/null"]        # 只保活
    command: ["simple-jev", "--model", "/models/Qwen3.5-0.8B", "--device", "cuda", "--dtype", "bfloat16", "--host", "0.0.0.0"]
```

> ⚠️ **必须显式传 `--host 0.0.0.0`**。上游服务的 `--host` 默认是 `127.0.0.1`，
> 只监听容器回环地址时，宿主机的 `-p 8000:8000` 是访问不到的。

服务就绪后可自检：

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/classifier -H 'Content-Type: application/json' \
  --data-binary '{"model":"/models/Qwen3.5-0.8B","state":"Mia owns a red bicycle.","questions":{"color":{"type":"choice","instructions":"What color is the bike?","criteria":{"red":null,"blue":null}}}}'
```

注意请求体里的 `model` 必须和启动时 `--model` 传的字符串完全一致。

### dtype 与显卡代际

上游 `--dtype` 默认 `bfloat16`，需要 **Ampere 及更新**的显卡（RTX 30 系 / A100 及以后）。
更老的卡（如 V100、T4）请改用 `--dtype float16`：

```bash
docker exec -it simple-jev simple-jev --model /models/Qwen3.5-0.8B --device cuda --dtype float16 --host 0.0.0.0
```

## 挂载模型

镜像里没有模型，三种用法任选：

| 场景 | 做法 |
| --- | --- |
| 本地已有模型目录 | `-v /host/model:/models`，然后 `--model /models/<名称>` |
| 用 Hugging Face 仓库名 | `-v hf-cache:/hf-cache`，然后 `--model Qwen/Qwen3.5-0.8B`（会联网下载到 `/hf-cache`） |
| 完全离线 | 先把权重放进 `/hf-cache` 卷，再加 `-e HF_HUB_OFFLINE=1` |

镜像内 `HF_HOME` 已固定为 `/hf-cache`，挂个卷上去就能让权重跨容器重启复用：

```bash
docker volume create hf-cache
docker run -d --name simple-jev --gpus all -p 8000:8000 \
  -v /宿主机/模型目录:/models \
  -v hf-cache:/hf-cache \
  ghcr.io/<owner>/simple-jev-docker:latest
```

## 本地构建

```bash
# 默认：CUDA 12.4 + torch cu124
docker build -t simple-jev:local .

# 固定上游版本，构建可复现的镜像
docker build --build-arg SIMPLE_JEV_REF=v0.1.0 -t simple-jev:v0.1.0 .
```

### 构建参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `BASE_IMAGE` | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | CUDA 基镜像。CUDA 12.4 **没有** ubuntu24.04 版本（官方从 12.5.1 才开始支持），所以是 22.04 |
| `PYTHON_VERSION` | `3.12` | Ubuntu 22.04 自带 3.10，低于上游 `requires-python >=3.12`，因此经 deadsnakes PPA 安装 3.12 |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu124` | PyTorch 的 wheel 索引，需与 `BASE_IMAGE` 的 CUDA 版本一致 |
| `TORCH_VERSION` | 空 | 指定 PyTorch 版本，例如 `2.6.0`。空 = 该索引上的最新版 |
| `SIMPLE_JEV_REPO` | `https://github.com/featherless-ai/simple-jev.git` | 上游仓库地址 |
| `SIMPLE_JEV_REF` | `main` | 分支 / tag / commit SHA，可固定版本 |
| `SIMPLE_JEV_EXTRAS` | 空 | 可选依赖，例如 `laya` |

> 📌 **cu124 索引上 PyTorch 最高只到 2.6.0**（正好等于上游 `torch>=2.6` 的下限）。
> 想要更新的 PyTorch，必须同时换掉基镜像和索引，例如
> `--build-arg BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126`
> —— 这两项必须成对修改，否则框架里的 CUDA 运行时会和基镜像不一致。

## GitHub Actions 工作流

`.github/workflows/docker-publish.yml`，构建环境为 `ubuntu-latest`（这里的 Ubuntu 指的是 **runner**，不是镜像本身）。

| 触发条件 | 行为 |
| --- | --- |
| push 到 `main` | 构建 + 推送，tag 为 `latest`、`main`、`sha-<短SHA>` |
| push `v*` tag | 构建 + 推送，tag 为 `latest`、`v1.2.3`、`1.2.3`、`1.2`、`sha-<短SHA>` |
| Pull Request | 只构建校验，**不推送** |
| 手动触发 | 可指定上游 ref 后构建 |

手动触发（Actions → Build and push Docker image → Run workflow）里 `SIMPLE_JEV_REF` 填分支/tag/SHA。

构建使用 Buildx + GitHub Actions 缓存（`type=gha`），重复构建会明显更快。

### 首次推送后：把 package 设为公开（否则匿名无法拉取）

GHCR 的 package 默认是私有。需要匿名 `docker pull` 时：
GitHub → 你的头像 → **Packages** → 选中本 package → **Package settings** → **Change visibility** → **Public**。

### 推送到其他镜像仓库（可选）

GHCR 在国内拉取可能较慢，如需同时推到 Docker Hub，在 workflow 的构建步骤后追加：

```yaml
      - name: Log in to Docker Hub
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Push to Docker Hub
        if: github.event_name != 'pull_request'
        uses: docker/build-push-action@v6
        with:
          context: .
          platforms: linux/amd64
          push: true
          tags: ${{ secrets.DOCKERHUB_USERNAME }}/simple-jev:latest
          build-args: |
            BASE_IMAGE=${{ env.BASE_IMAGE }}
            TORCH_INDEX_URL=${{ env.TORCH_INDEX_URL }}
            SIMPLE_JEV_REPO=${{ env.SIMPLE_JEV_REPO }}
            SIMPLE_JEV_REF=${{ env.SIMPLE_JEV_REF }}
          cache-from: type=gha
```

然后在仓库 Settings → Secrets and variables → Actions 里配置 `DOCKERHUB_USERNAME` 和 `DOCKERHUB_TOKEN`。
换成腾讯云 TCR / 阿里云 ACR 只需要改 `registry`、`username`、`password` 和镜像前缀。

## 说明与限制

- **仅 GPU / CUDA 12.4**：不提供 CPU 版本。CUDA 12.4 的镜像是 amd64 only。
- **ENTRYPOINT 已清空**：NVIDIA 基镜像自带 `/opt/nvidia/nvidia_entrypoint.sh`，本镜像用 `ENTRYPOINT []` 清掉了，
  这样 `docker run <镜像> <你的命令>` 能干净地整体替换默认命令。
- **`LD_LIBRARY_PATH` 保持基镜像的默认值**，没有覆盖 —— 覆盖会导致容器内找不到 CUDA 库。
- **镜像体积**：CUDA runtime 基镜像 + cu124 版 PyTorch，压缩后仍在数 GB 量级。
- **运行用户**：容器内以 root 运行，方便直接写挂载目录和访问 GPU。
- **无 HEALTHCHECK**：默认命令是 `tail`，容器探活会失败，所以没有配置探活；真正启动服务后可自行 `curl /health`。
- **无 VOLUME 声明**：避免 Docker 自动创建匿名卷，模型路径完全由你自己的 `-v` 决定。
- **首次启动很慢是正常的**：加载大模型权重需要时间，服务在权重加载完成后才开始监听。
- 本仓库不跟踪上游代码变化，上游更新后需重新构建镜像（手动触发 workflow 即可）。
