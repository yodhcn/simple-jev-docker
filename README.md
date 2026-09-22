# simple-jev Docker 镜像（CUDA 12.4）

把 [featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) 打包成可直接运行的 Docker 镜像，
由 GitHub Actions 在 Ubuntu runner 上构建，导出成 tar 后作为 **workflow artifact** 供你下载。

这个仓库**只**包含镜像定义（`Dockerfile` + workflow），不含上游源码 —— 构建时会自动把上游仓库克隆进镜像。
镜像**不推送到任何镜像仓库**（GHCR、Docker Hub 都不用），CI 的产物就是那个 tar.gz。

## 镜像的定位

| 项目 | 说明 |
| --- | --- |
| 🎯 目标环境 | **仅 GPU（CUDA 12.4）**，基于 `nvidia/cuda:12.4.1-runtime-ubuntu22.04`，不提供 CPU 版本 |
| 🏗️ 目标架构 | **仅 `linux/amd64`**（原因见[关于 arm64](#关于-arm64)） |
| ✅ 包含 | Python 3.12 + 独立 venv、PyTorch（cu124 构建）、Transformers / FastAPI 全量依赖、simple-jev 源码与 `simple-jev` 命令行入口 |
| ❌ 不包含 | **任何模型权重**。不预下载、不内置、不声明模型目录 |
| 🚀 默认命令 | `tail -f /dev/null` —— 容器起来后只是空转保活，服务由你自己启动 |
| 📦 端口 | 声明 `EXPOSE 8000`，是否映射由你决定 |
| 🔌 模型挂载 | 由你手动 `-v` 挂载，镜像不做任何预设 |

## 拿到镜像

镜像在 CI 里构建，产出一个 `docker save` 格式的 tar.gz。流程：

1. 仓库页面 → **Actions** → 左侧选 **`docker_image_build_artifact`** → **Run workflow**
2. 按需填参数（见[工作流参数](#工作流参数)），然后运行
3. 等构建完成（首次约 15–30 分钟），在该次运行页面底部 **Artifacts** 区域下载 `simple-jev-latest-amd64`
4. 解压后是三个文件：`.tar.gz`（镜像本体）、`.sha256`（校验和）、`.manifest.txt`（构建信息）

```bash
unzip simple-jev-latest-amd64.zip -d image && cd image

# 校验完整性
sha256sum -c simple-jev-latest-amd64.tar.gz.sha256

# 导入到本地 docker，得到 simple-jev:latest
docker load -i simple-jev-latest-amd64.tar.gz
docker images simple-jev
```

> ⚠️ artifact 默认保留 **30 天**后自动删除。这个 tar 是镜像的唯一副本（没有镜像仓库兜底），
> 建议下载后自己留存一份到服务器或对象存储。

前置条件：宿主机装好 NVIDIA 驱动和 nvidia-container-toolkit，且驱动支持 CUDA 12.4
（数据中心驱动 ≥ 470，消费级 Linux 驱动一般需 ≥ 525）。用 `nvidia-smi` 确认：

```bash
nvidia-smi                       # 右上角 "CUDA Version" 需 ≥ 12.4
docker run --rm --gpus all simple-jev:latest nvidia-smi
```

## 快速开始

以下命令里的镜像名统一用导入后得到的 `simple-jev:latest`。

### 1. 启动容器并保持空转

```bash
docker run -d --name simple-jev --gpus all \
  -p 8000:8000 \
  -v /宿主机/模型目录:/models \
  simple-jev:latest
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
  simple-jev:latest \
  simple-jev --model /models/Qwen3.5-0.8B --device cuda --dtype bfloat16 --host 0.0.0.0 --port 8000
```

**方式 C：docker compose**，用 `command:` 覆盖：

```yaml
services:
  simple-jev:
    image: simple-jev:latest
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
  simple-jev:latest
```

## GitHub Actions 工作流

`.github/workflows/docker-build-artifact.yml`。构建环境为 `ubuntu-latest`（这里的 Ubuntu 指的是 **runner**，不是镜像本身）。

**只支持手动触发**：Actions → `docker_image_build_artifact` → Run workflow。
没有 push / PR 触发，所以合并代码不会自动构建。

### 工作流参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `image_tag` | `latest` | 给镜像打的 tag，决定 tar 里的镜像名 `simple-jev:<tag>` |
| `simple_jev_ref` | `main` | 上游 simple-jev 的 ref：分支 / tag / commit SHA。要可复现的镜像就固定成 tag 或 SHA |
| `torch_version` | 空 | 指定 PyTorch 版本（例如 `2.6.0`）。空 = cu124 索引上的最新版 |
| `simple_jev_extras` | 空 | 上游 hf-server 的可选依赖，例如 `laya` |
| `cleanup` | `true` | 构建前清理 runner 磁盘。镜像约 7 GB，而 hosted runner 默认只剩 ~14 GB 可用，**建议保持开启** |

构建参数 `BASE_IMAGE` 和 `TORCH_INDEX_URL` 由 `Dockerfile` 决定，workflow 不再重复传，
避免两处漂移；要改 CUDA / PyTorch 版本直接改 `Dockerfile` 的 `ARG` 默认值。

产物：

- artifact 名 `simple-jev-<tag>-amd64`，内含 `.tar.gz` + `.sha256` + `.manifest.txt`
- `.manifest.txt` 里记了镜像 ID、大小、tar 的 sha256、内嵌的 torch / CUDA 版本、上游 ref 和产生它的那次 run，方便日后追溯
- 上传时用 `compression-level: 0`（tar.gz 已经压过一遍，再压纯属浪费 CPU 和上传时间）
- 保留 30 天，同一 tag 重跑会直接覆盖旧 artifact

构建使用 Buildx + GitHub Actions 缓存（`type=gha`），CUDA / PyTorch 那两层能被缓存住，
重复构建省掉约 5 GB 下载。注意 Actions 缓存每个仓库上限 10 GB，镜像偏大时缓存可能被挤掉，属正常现象。

## 本地构建

没有 CI 或者想快速试错时可以本地构建（本仓库的 `.dockerignore` 只放行 `Dockerfile`，
构建上下文只有几 KB，因为上游源码是在镜像里 clone 的）：

```bash
# 默认：CUDA 12.4 + torch cu124
docker build -t simple-jev:local .

# 固定上游版本，构建可复现的镜像
docker build --build-arg SIMPLE_JEV_REF=v0.1.0 -t simple-jev:v0.1.0 .

# 构建完自己导出成同样的 tar（和 CI 的产物格式一致）
docker save simple-jev:v0.1.0 | pigz > simple-jev-v0.1.0.tar.gz
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
> 镜像构建期有一条断言 `torch.version.cuda == '12.4'`，装错版本会直接构建失败而不是静默出货。

## 关于 arm64

当前 workflow 锁死 `linux/amd64`，不是偷懒，是这条技术栈在 arm64 上确实走不通：

1. **cu124 索引没有满足 `torch>=2.6` 的 arm64 wheel。** 该索引上 aarch64 的包只到 2.5.x，
   而 2.6.0 只有 `linux_x86_64` / `win_amd64`；上游要求 `torch>=2.6`，pip 会直接解析失败。
2. CUDA 12.4 的官方镜像也没有 Ubuntu 24.04 版本（Ubuntu 系只有 20.04 / 22.04）。

真需要 arm64 的话，要一次性改三处：

| 位置 | 改成 |
| --- | --- |
| `Dockerfile` 的 `BASE_IMAGE` | 换到有 arm64 标签的 CUDA 镜像（12.5.1 起才提供 ubuntu24.04，arm64 标签需自行到 Docker Hub 确认） |
| `Dockerfile` 的 `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu126` —— 该索引上有 `torch-2.6.0+cu126-cp312-cp312-linux_aarch64.whl` |
| `Dockerfile` 的构建期断言 | `torch.version.cuda == '12.4'` 要跟着改成 `'12.6'` |
| workflow 的 `runs-on` / `PLATFORM` | `ubuntu-24.04-arm`（**仅公共仓库免费**）+ `linux/arm64` |

## 说明与限制

- **仅 GPU / CUDA 12.4**：不提供 CPU 版本。CUDA 12.4 的镜像是 amd64 only。
- **不推送镜像仓库**：CI 的产物只有 artifact，仓库里没有镜像副本，所以 artifact 过期前记得下载留存。
- **ENTRYPOINT 已清空**：NVIDIA 基镜像自带 `/opt/nvidia/nvidia_entrypoint.sh`，本镜像用 `ENTRYPOINT []` 清掉了，
  这样 `docker run <镜像> <你的命令>` 能干净地整体替换默认命令（同理 `docker exec` 时 `simple-jev` 直接可用）。
- **`LD_LIBRARY_PATH` 保持基镜像的默认值**，没有覆盖 —— 覆盖会导致容器内找不到 CUDA 库。
- **镜像体积**：CUDA runtime 基镜像 + cu124 版 PyTorch，`docker save` 出的 tar 未压缩约 7 GB 量级，
  压缩后约 3 GB；所以 workflow 里会先 gzip 再上传。
- **单 artifact 上限 10 GB**：这是 GitHub 的硬限制、不可调，因此镜像必须压缩后再传。
- **运行用户**：容器内以 root 运行，方便直接写挂载目录和访问 GPU。
- **无 HEALTHCHECK**：默认命令是 `tail`，容器探活会失败，所以没有配置探活；真正启动服务后可自行 `curl /health`。
- **无 VOLUME 声明**：避免 Docker 自动创建匿名卷，模型路径完全由你自己的 `-v` 决定。
- **首次启动很慢是正常的**：加载大模型权重需要时间，服务在权重加载完成后才开始监听。
- 本仓库不跟踪上游代码变化，上游更新后需重新构建镜像（手动触发 workflow 即可）。
