# kev Docker 镜像

把 [jaredpalmer/kev](https://github.com/jaredpalmer/kev) 打包成可直接运行的 Docker 镜像，
由 GitHub Actions 在 Ubuntu runner 上构建，导出成 tar 后作为 **workflow artifact** 供你下载。

这个仓库**只**包含镜像定义（`Dockerfile` + `kev-serve` + workflow），不含上游源码 —— 构建时会自动把上游仓库克隆进镜像。
镜像**不推送到任何镜像仓库**（GHCR、Docker Hub 都不用），CI 的产物就是那个 tar.gz。

> kev 是一族基于 Qwen3.5 的小型决策模型（LoRA 适配器 + 指针头），对外提供与 TypeSafe
> System One 兼容的 `/v1/systemone` 接口，可以回答 `noul`（是否）、`choice`（多选）、`score`（评分）三类问题。

## 镜像的定位

| 项目 | 说明 |
| --- | --- |
| 🎯 目标环境 | **仅 GPU**，基镜像 `nvidia/cuda:12.4.1-runtime-ubuntu22.04`，不提供 CPU 版本 |
| 🏗️ 目标架构 | **仅 `linux/amd64`** |
| 🐍 Python | **3.13**（与上游 `.python-version` 一致），Ubuntu 22.04 自带 3.10，故经 deadsnakes PPA 安装 |
| 📦 依赖 | 用 `uv sync --frozen` 按上游 **`uv.lock`** 安装，版本完全由锁文件决定 |
| ✅ 包含 | Python 3.13 + uv 虚拟环境、PyTorch、Transformers / PEFT、FastAPI 服务端、kev 源码本体 |
| ❌ 不包含 | **任何模型权重**。不预下载、不内置、不声明模型目录 |
| 🚀 默认命令 | `tail -f /dev/null` —— 容器起来后只是空转保活，服务由你自己启动 |
| 📦 端口 | 声明 `EXPOSE 8009`，是否映射由你决定 |
| 🔌 模型挂载 | 由你手动 `-v` 挂载，镜像不做任何预设 |

## 拿到镜像

镜像在 CI 里构建，产出一个 `docker save` 格式的 tar.gz。流程：

1. 仓库页面 → **Actions** → 左侧选 **`docker_image_build_artifact`** → **Run workflow**
2. 按需填参数（见[工作流参数](#工作流参数)），然后运行
3. 等构建完成（首次约 15–30 分钟），在该次运行页面底部 **Artifacts** 区域下载 `kev-latest-amd64`
4. 解压后是三个文件：`.tar.gz`（镜像本体）、`.sha256`（校验和）、`.manifest.txt`（构建信息）

```bash
unzip kev-latest-amd64.zip -d image && cd image

# 校验完整性
sha256sum -c kev-latest-amd64.tar.gz.sha256

# 导入到本地 docker，得到 kev:latest
docker load -i kev-latest-amd64.tar.gz
docker images kev
```

> ⚠️ artifact 默认保留 **30 天**后自动删除。这个 tar 是镜像的唯一副本（没有镜像仓库兜底），
> 建议下载后自己留存一份到服务器或对象存储。

前置条件：宿主机装好 NVIDIA 驱动和 nvidia-container-toolkit。驱动版本要求见
[CUDA 版本与基镜像](#cuda-版本与基镜像) —— 简而言之**驱动 ≥ 525**。

```bash
nvidia-smi                       # 确认驱动与 GPU 可见
docker run --rm --gpus all kev:latest nvidia-smi
```

## 模型准备（重要）

kev 的一个 checkpoint 是**一个目录**，里面至少要有：

```
head.pt                     # 元数据（含基座模型名、LoRA rank、温度等）+ 指针头权重
adapter_config.json         # LoRA 配置
adapter_model.safetensors   # LoRA 适配器权重
tokenizer 相关文件
```

**关键点：checkpoint 里只存了基座模型的*名字*，不存基座权重。** `head.pt` 里的 `base` / `base_revision`
两个字段声明了它训练时用的基座 —— 以 `jaredpalmer/kev-4b` 为例（解开该文件实测）：

| 字段 | 值 |
| --- | --- |
| `base` | `Qwen/Qwen3.5-4B-Base` |
| `base_revision` | `1001bb4d826a52d1f399e183466143f4da7b741b` |

`kev/checkpoint.py` 读出这两个字段后交给 transformers，经 Hugging Face 缓存解析
（`load_tokenizer(meta.base, revision=meta.base_revision)` 和 `DecisionModel(meta.base, ..., revision=...)`）。
**上游代码里并没有写死这个模型名** —— 值是 checkpoint 自己带的，`kev/transfer_v9.py:35` 里那份
`Qwen/Qwen3.5-4B-Base → 1001bb4d` 的映射只服务于评测数据生成，评测/训练命令（`README.md:264`）里那个
`--base_revision` 也是训练参数，服务路径都不经过它们。

所以有三种用法：

| 场景 | 做法 | 是否联网 |
| --- | --- | --- |
| 首次省事 | checkpoint 挂进容器，**基座走 HF 缓存**（`-v kev-hf-cache:/hf-cache`） | 首次需要联网下载基座 |
| 完全离线 | 宿主机备好 HF 缓存目录，**整个挂进 `/hf-cache`**，再加 `-e HF_HUB_OFFLINE=1` | 不需要 |
| 基座自己指定 | 基座权重放宿主机目录，挂进来并用 **`--base-model` 覆盖**（见下） | 不需要 |

第三种才是「模型路径从外部挂载」的完整形态 —— 连基座也不经过 HF 缓存：

```bash
docker run -d --name kev --gpus all \
  -p 8009:8009 \
  -v /data/kev-4b:/models/kev-4b \
  -v /data/qwen3.5-4b-base:/models/qwen3.5-4b-base \
  kev:latest \
  kev-serve --run /models/kev-4b --port 8009 \
            --base-model /models/qwen3.5-4b-base
```

`--base-model` 既接受**本地目录**也接受 **Hub 仓库 ID**（`--base-revision` 可进一步钉住分支/tag/SHA）。
指向本地目录时会自动把 `base_revision` 置空 —— revision 只对 Hub 仓库有意义，本地快照上留着 checkpoint
那个 sha 只是把一个无意义的参数递给 transformers。

也可以把 checkpoint 的位置交给 HF 自己管：`--run jaredpalmer/kev-4b` 传 Hub 仓库 ID（支持
`jaredpalmer/kev-4b@qwen3` 这种 `@revision` 写法），checkpoint 会下载到 `/hf-cache`。

获取 checkpoint 的两种来源：Hugging Face 上的 `jaredpalmer/kev-0.8b` / `kev-4b` / `kev-9b`，
或上游 [GitHub release](https://github.com/jaredpalmer/kev/releases/tag/kev-family) 的 tarball（带 SHA-256 校验和）。

```bash
# 把 checkpoint 放到宿主机上，比如 /data/kev-4b
ls /data/kev-4b          # head.pt  adapter_config.json  adapter_model.safetensors  ...
```

## 快速开始

以下命令里的镜像名统一用导入后得到的 `kev:latest`。

### 1. 启动容器并保持空转

```bash
docker volume create kev-hf-cache

docker run -d --name kev --gpus all \
  -p 8009:8009 \
  -v /data/kev-4b:/models/kev-4b \
  -v kev-hf-cache:/hf-cache \
  kev:latest
```

此时容器内只有 `tail -f /dev/null` 在跑，模型和服务都没启动。

### 2. 自己启动服务

用镜像里的 **`kev-serve`**（而不是 `python -m kev.serve`）：

```bash
docker exec -it kev kev-serve --run /models/kev-4b --port 8009
```

**方式 B：启动时直接替换命令** —— 不保留空转：

```bash
docker run --rm --name kev --gpus all -p 8009:8009 \
  -v /data/kev-4b:/models/kev-4b \
  -v kev-hf-cache:/hf-cache \
  kev:latest \
  kev-serve --run /models/kev-4b --port 8009
```

要连基座也完全走本地挂载，就在末尾追加 `--base-model /models/qwen3.5-4b-base` 并多挂一个卷，
完整命令见上一节「模型准备」。

**方式 C：docker compose**，用 `command:` 覆盖：

```yaml
services:
  kev:
    image: kev:latest
    container_name: kev
    ports:
      - "8009:8009"
    volumes:
      - /data/kev-4b:/models/kev-4b
      - kev-hf-cache:/hf-cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    # command: ["tail", "-f", "/dev/null"]           # 只保活
    command: ["kev-serve", "--run", "/models/kev-4b", "--port", "8009"]
```

> ⚠️ **`--run` 必须指向真实存在的目录。** 上游 `serve.py` 在路径不存在时会静默回退到
> `--fallback`（默认 `runs/smoke`），而 `runs/smoke` 看起来像个 Hub 仓库 ID，
> 于是报出一个"仓库不存在"的错误 —— 与实际原因（路径写错）完全无关。看到这类报错先检查挂载路径。

### 3. 调用

服务就绪后（加载权重期间不监听端口，首次可能要等一会儿）：

```bash
curl -s localhost:8009/v1/models

curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "model": "kev-latest",
  "questions": {
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                 "shipping": "Delivery status, delays, lost packages",
                                 "billing": "Charges, invoices, payment problems"}},
    "escalate":    {"type": "noul",  "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }}'
```

请求体里的 `model` 字段是**回显**用的，任意字符串都行（上游示例统一用 `kev-latest`）。

其他端点：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/v1/models` | 模型卡片 + 当前加载的 checkpoint、设备、dtype、温度、前缀缓存命中统计 |
| `POST` | `/v1/systemone` | 主接口 |
| `POST` | `/v1/systemone/permute` | 同一个 Choice 问题换多种选项顺序各跑一遍，看答案是否稳定 |
| `POST` | `/v1/systemone/separate` | 每个问题各跑一次前向（用于对比"打包提问 vs 分开提问"） |

## `kev-serve`：本仓库对上游的两处包装

`kev-serve` 是**本仓库唯一的运行时改动**，它只做两件事，其余全部沿用上游（设备选择、`LoadOptions.from_env`、
CUDA 上的 bf16 默认值、`kev.serve` 的整个参数面）。

### ① bind 地址

上游 `kev/serve.py` 最后一行是：

```python
uvicorn.run(app, host="127.0.0.1", port=a.port)
```

**host 是硬编码的 `127.0.0.1`，而且 argparse 只暴露了 `--run` / `--fallback` / `--port`，没有 `--host`。**
在笔记本上没问题，但在容器里这意味着服务只监听回环地址，Docker 的 `-p 8009:8009` 是 DNAT 到容器的
eth0 地址上的，永远打不通 —— 而且**两端都不报错**，属于静默失败。

实现方式是在 `kev.serve.main()` 之前替换 `uvicorn.run`，而不是复制一份 `main()` —— 复制会随上游改动悄悄腐烂。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `KEV_HOST` | `0.0.0.0` | bind 地址。设为空字符串则退回上游的 `127.0.0.1`；设成具体网卡地址可收窄暴露面 |
| `KEV_PORT` | 未设置 | 优先于上游的 `--port`，方便 compose / `.env` 管理 |

### ② 基座模型来源

如上一节所述，基座由 checkpoint 的 `head.pt` 声明，上游没有覆盖入口。`kev-serve` 补上两个参数，
在读完 `head.pt` 之后改写 `Meta.base` / `Meta.base_revision`：

| CLI | 环境变量 | 说明 |
| --- | --- | --- |
| `--base-model PATH_OR_REPO_ID` | `KEV_BASE_MODEL` | 覆盖基座：本地目录或 Hub 仓库 ID。不传＝沿用 checkpoint 声明的值 |
| `--base-revision REV` | `KEV_BASE_REVISION` | 钉住分支 / tag / commit SHA。传空字符串 `""` = 不钉（用仓库默认版本） |

优先级 **CLI > 环境变量 > checkpoint**。指向本地目录且未显式给 `--base-revision` 时，revision 自动置空。
不传任何一项时行为与上游完全一致（有测试覆盖这一条）。

上游的 `--help` 由上游 parser 负责打印，看不到这两个参数，所以 `kev-serve --help` 会先补一段自己的说明：

```bash
docker exec -it kev kev-serve --help
```

为什么改 `Checkpoint.__init__` 而不是 `load`：这样一次性覆盖了所有读取该字段的地方 —— `load`
（tokenizer + DecisionModel）、`hybrid_base()`（用 `AutoConfig` 判断是否混合骨干，进而决定 MLX 后端）、
以及 `/v1/models` 返回的模型卡。最后那处正好也是**验证覆盖是否生效**的地方：

```bash
curl -s localhost:8009/v1/models | python -m json.tool | grep '"base"'
```

启动时如果覆盖生效，日志里会多一行 `kev-serve: base model Qwen/Qwen3.5-4B-Base@1001bb4d... -> ...`。

**不想用这个包装层的话**，也可以让上游原样跑，代价是要放弃端口映射、改用 host 网络，且无法覆盖基座：

```bash
docker run --rm --network host --gpus all \
  -v /data/kev-4b:/models/kev-4b -v kev-hf-cache:/hf-cache \
  kev:latest \
  python -m kev.serve --run /models/kev-4b --port 8009
# --network host 下容器内的 127.0.0.1 就是宿主机回环，所以能访问，但端口隔离也没了
```

反过来说，只想临时验证 shim 没加私货：`docker exec -it kev python -m kev.serve --help`。

## 常用环境变量

除了上面的 `KEV_HOST` / `KEV_PORT` / `KEV_BASE_MODEL` / `KEV_BASE_REVISION`，上游还认这些（都是运行时 `-e` 传）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `KEV_DTYPE` | CUDA 上 `bf16` | `bf16` / `fp16` / `fp32`。`fp32` 是上游所有已公布数字所用的"精确路径"，更慢但更可复现 |
| `KEV_MERGE` | `1` | 是否把 LoRA 合进基座权重。`0` 关闭 |
| `KEV_ATTN` | 模型默认 | 注意力后端，如 `sdpa` / `eager` |
| `KEV_LORA_SCALE` | `1` | WiSE-FT 式插值，在基座（0）与微调权重（1）之间 |
| `KEV_TEMPERATURE` | checkpoint 自带 | `1.0` = 原始 logits，不做校准 |
| `KEV_BACKEND` | `torch` | `torch` / `mlx` / `auto`（MLX 是 Apple Silicon 专用，容器里用不上） |
| `KEV_DATE_FACTS` | `0` | `1` 则在文本里追加日期差（kev 自己算不准日期，但能用现成的天数） |
| `KEV_API_KEY` | 未设置 | 设置后 `/v1/*` 要求 `Authorization: Bearer <key>` |
| `KEV_PREFIX_CACHE` | `4` | 状态前缀缓存条数，`0` 关闭 |
| `KEV_PREFIX_MIN_TOKENS` | `384` | 状态短于该值时不做前缀分离 |
| `HF_HOME` | `/hf-cache` | 权重缓存位置（镜像内已固定，挂卷即可复用） |
| `HF_HUB_OFFLINE` | 未设置 | `1` 强制离线，权重必须已在缓存里 |

## CUDA 版本与基镜像

这里有个容易误解的点，值得单独说明：**基镜像上的 `12.4` 并不代表 torch 实际跑的 CUDA。**

上游的 `uv.lock` 把 torch 解析成 **PyPI 上的 2.8.0**（不是 PyTorch 官方 CUDA 索引里的变体）。
PyPI 的 Linux torch wheel 会**自带一整套 CUDA 用户态库**（以 `nvidia-*-cu12` 依赖的形式，
锁文件里能看到 `nvidia-cuda-runtime-cu12` 等），torch 运行时优先加载的就是这一套，
而不是基镜像里的 CUDA。所以基镜像在这里主要提供 CUDA 的工具链布局和驱动挂载点。

实际约束落在**宿主机驱动**上：CUDA 12.x 的次版本兼容规则允许 12.8 的用户态库跑在
**驱动 ≥ 525** 上。想要 CUDA 12.8 的完整特性则建议驱动 ≥ 570。

如果你更希望基镜像和 torch 的 CUDA 版本严格一致，把基镜像换成 12.8 即可（改一处）：

```dockerfile
ARG BASE_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu22.04
```

**为什么不用 PyTorch 官方的 cuXXX 索引**：`uv sync --frozen` 完全按 `uv.lock` 安装，
而锁文件里记录的就是 PyPI 的 wheel 及其哈希。换索引等于绕开锁文件重新解析，
会同时丢掉版本可复现性和哈希校验 —— 为了对齐一个本身不参与运算的版本号，不划算。

## flash-linear-attention

Qwen3.5 的骨干混了 Gated DeltaNet 层。`transformers` 在这几层上用的是
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) 的 Triton 内核，
取用方式是在建模代码里挂装饰器 `use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")`
（已对着 pinned 的 transformers 5.17.0 源码核实）。它会 `import fla`，再解析
`fla.ops.gated_delta_rule.chunk_gated_delta_rule`；**解析不到时不抛异常**，只打一条 warning
然后退回参考 PyTorch 实现 —— 官方注释原话是 "This is correct but much slower"，量级差一个数量级。

所以这里要分清两件事：**装了**（`import fla` 成功）和**生效了**（那个符号真的解析到）。
`Dockerfile` 的构建期自检断的是后者，因为前者在出错的那个场景里照样通过。

它不在 `uv.lock` 里，由 `Dockerfile` 单独 `uv pip install`，**按上游自己的 Modal 配方装不带 extra 的包名**：

- `flash-linear-attention`（bare）的依赖只有 `fla-core` + `einops` + `transformers>=4.45`，
  都不带 torch / triton 约束，所以这一步**不可能把 torch 或 triton 拉出 `uv.lock`**
- 早先写的 `flash-linear-attention[cuda]` 会额外引入 `torch>=2.7` / `triton>=3.3`。这些约束今天是被满足的，
  但留着等于给未来某个 fla 版本一个改 torch 的机会，因此改回 bare
- `einops` 是这一步唯一一个 `uv.lock` 里没有的包（`fla-core` 无条件依赖它）

> **和上游 Modal 镜像的两处有意差异**
>
> 1. **triton 保持锁里的 3.4.0。** 上游额外强制 `triton>=3.7.1`，只为绕开 gated-chunk 的
>    **反向传播** bug（fla#640）。本镜像只跑前向推理，不需要这个 workaround，也就不去动锁里的 pin。
> 2. **不装 `causal-conv1d`**（上游也没装），所以短卷积退回 `F.conv1d`。相对 delta-rule 内核本身，
>    这是次要开销。

不想要它的话，构建时传 `--build-arg INSTALL_FLASH_LINEAR_ATTENTION=0`（workflow 里对应 `flash_linear_attention` 开关）。
需要两次构建完全一致时，用 `--build-arg FLASH_LINEAR_ATTENTION_SPEC=flash-linear-attention==0.5.2` 把它也钉住
（默认不钉，与上游一致）。

确认运行中的容器里内核确实生效（输出应以 `fla.` 开头；若报 ImportError，说明装的那份没解析到）：

```bash
docker exec kev python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule as f; print(f.__module__)"
```

## GitHub Actions 工作流

`.github/workflows/docker-build-artifact.yml`。构建环境为 `ubuntu-latest`（这里的 Ubuntu 指的是 **runner**，不是镜像本身）。

**只支持手动触发**：Actions → `docker_image_build_artifact` → Run workflow。
没有 push / PR 触发，所以合并代码不会自动构建。

### 工作流参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `image_tag` | `latest` | 给镜像打的 tag，决定 tar 里的镜像名 `kev:<tag>` |
| `kev_ref` | `main` | 上游 kev 的 ref：分支 / tag / commit SHA。要可复现的镜像就固定成 tag 或 SHA |
| `flash_linear_attention` | `true` | 是否安装 Gated DeltaNet 内核（见上一节） |
| `cleanup` | `true` | 构建前清理 runner 磁盘。hosted runner 默认空间紧张，**建议保持开启** |

基镜像、Python 版本、uv 版本、依赖版本都由 `Dockerfile` / `uv.lock` 决定，workflow 不再重复传，
避免两处漂移。

产物：

- artifact 名 `kev-<tag>-amd64`，内含 `.tar.gz` + `.sha256` + `.manifest.txt`
- `.manifest.txt` 里记了镜像 ID、大小、tar 的 sha256、镜像内实际的 Python / torch / CUDA 版本、上游 ref
  和产生它的那次 run，方便日后追溯
- 上传时用 `compression-level: 0`（tar.gz 已经压过一遍，再压纯属浪费 CPU 和上传时间）
- 保留 30 天，同一 tag 重跑会直接覆盖旧 artifact

构建使用 Buildx + GitHub Actions 缓存（`type=gha`），CUDA / torch 那两层能被缓存住，重复构建省掉数 GB 下载。
注意 Actions 缓存每个仓库上限 10 GB，镜像偏大时缓存可能被挤掉，属正常现象。

## 本地构建

没有 CI 或者想快速试错时可以本地构建（`.dockerignore` 只放行 `Dockerfile` 和 `kev-serve`，
构建上下文只有几 KB，因为上游源码是在镜像里 clone 的）：

```bash
# 默认：CUDA 12.4 基镜像 + 上游 uv.lock
docker build -t kev:local .

# 固定上游版本，构建可复现的镜像
docker build --build-arg KEV_REF=kev-family -t kev:kev-family .

# 构建完自己导出成同样的 tar（和 CI 的产物格式一致）
docker save kev:local | pigz > kev-local.tar.gz
```

### 构建参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `BASE_IMAGE` | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | CUDA 基镜像。CUDA 12.4 **没有** ubuntu24.04 版本（官方从 12.5.1 才开始支持），所以是 22.04 |
| `PYTHON_VERSION` | `3.13` | 与上游 `.python-version` 一致。Ubuntu 22.04 自带 3.10，因此经 deadsnakes PPA 安装 |
| `UV_VERSION` | `0.12.18` | 读 `uv.lock` 的工具版本 |
| `KEV_REPO` | `https://github.com/jaredpalmer/kev.git` | 上游仓库地址 |
| `KEV_REF` | `main` | 分支 / tag / commit SHA，可固定版本 |
| `INSTALL_FLASH_LINEAR_ATTENTION` | `1` | 是否安装 DeltaNet 内核 |
| `FLASH_LINEAR_ATTENTION_SPEC` | `flash-linear-attention` | 上一项具体装什么。默认不钉版本（与上游一致），钉住才能让两次构建完全一致 |

## 说明与限制

- **仅 GPU**：不提供 CPU 版本。
- **不推送镜像仓库**：CI 的产物只有 artifact，仓库里没有镜像副本，所以 artifact 过期前记得下载留存。
- **不内置模型**：镜像里没有任何权重，checkpoint 和基座都靠 `-v` 挂载或 HF 缓存。
- **驱动要求**：≥ 525（torch 自带的 CUDA 用户态库是 12.8，受 CUDA 12.x 次版本兼容规则约束）。
- **`kev-serve` 是本仓库唯一对上游的改动**，只做两件事：改 bind 地址、允许覆盖基座模型（`--base-model` /
  `KEV_BASE_MODEL`）。上游源码本身未被修改，两条都在 `kev.checkpoint` 已有的接缝上完成。
- **ENTRYPOINT 已清空**：NVIDIA 基镜像自带 `/opt/nvidia/nvidia_entrypoint.sh`，本镜像用 `ENTRYPOINT []` 清掉了，
  这样 `docker run <镜像> <你的命令>` 能干净地整体替换默认命令。
- **`LD_LIBRARY_PATH` 保持基镜像的默认值**，没有覆盖 —— 覆盖会导致容器内找不到 CUDA 库。
- **镜像体积**：CUDA 基镜像 + 自带 CUDA 库的 torch wheel + 构建工具链，`docker save` 出的 tar 未压缩约 8 GB 量级，
  压缩后约 3–4 GB。**单 artifact 上限 10 GB** 是 GitHub 的硬限制、不可调，因此镜像必须压缩后再传 ——
  如果哪天镜像涨到压完还超过 10 GB，就得先把 `BASE_IMAGE` 换成更瘦的变体，或改用镜像仓库。
- **`build-essential` 装在镜像里**：`uv.lock` 同时记录 wheel 和 sdist，万一某个包没有 cp313/linux 的 wheel，
  uv 需要现场编译。这是为了构建可靠性做的取舍，代价约 250 MB。
- **串行推理**：上游服务一次只处理一个请求，靠状态前缀缓存加速重复文本，但不会对不同调用方做批处理。
- **运行用户**：容器内以 root 运行，方便直接写挂载目录和访问 GPU。
- **无 HEALTHCHECK**：默认命令是 `tail`，容器探活会失败，所以没有配置探活；真正启动服务后可自行 `curl /v1/models`。
- **无 VOLUME 声明**：避免 Docker 自动创建匿名卷，模型路径完全由你自己的 `-v` 决定。
- **首次启动很慢是正常的**：要先下载（或从缓存加载）基座模型，再套 LoRA 和指针头，期间不监听端口。
- 本仓库不跟踪上游代码变化，上游更新后需重新构建镜像（手动触发 workflow 即可）。
