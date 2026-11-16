# 数据/模型下载指南

本目录的脚本用于从 ModelScope 或 HuggingFace 下载数据集和模型，适配本集群的网络环境。

---

## 集群网络情况（2026-10 实测）

| 域名 | 状态 | 备注 |
|---|---|---|
| `www.modelscope.cn` | 可用，延迟约 0.26 秒，速度约 **200 MB/s** | SDK 默认指向 `modelscope.cn`（不带 www），该地址超时 |
| `modelscope.cn` | 不通 | DNS 可解析，但 TCP 连接超时 |
| `huggingface.co` | 不通 | DNS 返回假 IP（199.59.148.7） |
| `hf-mirror.com` | 可用，速度约 **2-40 MB/s** | 证书为第三方 CA 签发，curl 默认不认；Python 的 certifi 可以正常校验 |
| `github.com` | 可用 | |

**结论**：
- 优先用 ModelScope，速度快，稳定。
- HuggingFace 走 `hf-mirror.com`，速度不稳定，大文件下载偶发超时，靠重试续传完成。
- 两个 SDK 都在 `areal2` conda 环境里（`modelscope 1.40.2`，`huggingface_hub 1.29.0`）。

---

## 文件说明

| 文件 | 说明 |
|---|---|
| `download.py` | 统一下载脚本，支持 ModelScope / HuggingFace，模型和数据集 |
| `download_data.py` | 早期版本，仅支持 ModelScope 数据集，保留备用 |
| `download.sh` | conda `areal2` 环境安装脚本，与下载无关 |

---

## 快速开始

```bash
cd /home/admin/chentinhong.cth/inclusionAI/AReaL/download
conda activate areal2
```

**下载模型（ModelScope，默认）**

```bash
python download.py --repo Qwen/Qwen2.5-7B-Instruct \
    --local-dir /storage/openpsi/users/chen/code_data/models/Qwen2.5-7B-Instruct
```

**下载数据集（ModelScope）**

```bash
python download.py --repo AI-ModelScope/gsm8k --type dataset \
    --local-dir /storage/openpsi/users/chen/code_data/datasets/gsm8k
```

**下载模型（HuggingFace，走 hf-mirror.com）**

```bash
python download.py --source hf --repo Qwen/Qwen2.5-7B-Instruct \
    --local-dir /storage/openpsi/users/chen/code_data/models/Qwen2.5-7B-Instruct
```

**下载数据集（HuggingFace）**

```bash
python download.py --source hf --repo openai/gsm8k --type dataset \
    --include "main/*" \
    --local-dir /storage/openpsi/users/chen/code_data/datasets/gsm8k
```

**只下部分文件**

```bash
python download.py --repo Qwen/Qwen2.5-7B-Instruct \
    --include "*.json" "*.safetensors" \
    --local-dir /storage/openpsi/users/chen/code_data/models/Qwen2.5-7B-Instruct
```

**使用 `--subdir` 快捷路径**（自动落在 `code_data/<models|datasets>/<名字>`）

```bash
python download.py --repo AI-ModelScope/gsm8k --type dataset --subdir
# 等同于 --local-dir /storage/openpsi/users/chen/code_data/datasets/gsm8k
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--repo` | 必填 | 仓库 ID，如 `Qwen/Qwen2.5-7B-Instruct` |
| `--source` | `modelscope` | `modelscope` / `ms` / `huggingface` / `hf` |
| `--type` | `model` | `model` 或 `dataset` |
| `--local-dir` | 无（必填，或用 `--subdir`） | 下载目标目录 |
| `--subdir` | false | 自动构造路径为 `<data-root>/<type>s/<repo名>` |
| `--data-root` | `/storage/openpsi/users/chen/code_data` | `--subdir` 使用的根目录 |
| `--revision` | 无 | 指定 branch/tag/commit |
| `--include` | 无 | glob 白名单，如 `"*.safetensors"` |
| `--exclude` | 无 | glob 黑名单 |
| `--max-workers` | `8` | 并发下载线程数 |
| `--retries` | `3` | 网络失败时的重试次数，两个 SDK 都支持断点续传 |

`--local-dir` 和 `--subdir` 必须提供其中一个，否则脚本报错退出（exit 2），不会静默落到默认路径。

---

## 大文件下载建议

大文件（> 1 GB）建议用 `nohup` 后台运行并把日志写到文件：

```bash
nohup python download.py \
    --source hf \
    --repo shunzou05/TerminalHorizon-Environment \
    --type dataset \
    --local-dir /storage/openpsi/users/chen/code_data/datasets/TerminalHorizon-Environment \
    --retries 10 \
    > /storage/openpsi/users/chen/code_data/download_TerminalHorizon.log 2>&1 &

echo "PID=$!"
tail -f /storage/openpsi/users/chen/code_data/download_TerminalHorizon.log
```

中断后重新运行同一条命令会自动续传，已下载的文件不会重复下载。

---

## 环境变量说明

`download.py` 在启动时用 `os.environ.setdefault` 设置以下变量，无需手动 export：

| 变量 | 值 | 原因 |
|---|---|---|
| `MODELSCOPE_ENDPOINT` | `https://www.modelscope.cn` | SDK 默认的 `modelscope.cn` 在本集群超时 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | `huggingface.co` 在本集群 DNS 被污染 |
| `HF_HUB_DISABLE_XET` | `1` | xet 传输协议会被重置（Connection reset by peer） |
| `HF_HUB_ETAG_TIMEOUT` | `60` | 默认 10 秒在首次请求时经常超时 |
| `HF_HUB_DOWNLOAD_TIMEOUT` | `120` | 防止大文件下载中途超时 |

如果需要覆盖，在运行前 export 对应变量即可。

---

## 私有/限制访问仓库

```bash
export MODELSCOPE_API_TOKEN=your_token   # ModelScope
export HF_TOKEN=your_token               # HuggingFace
```

不要在命令行里传 token。hf-mirror.com 是第三方站点，不要在这条链路上使用包含敏感权限的 HF token。

---

## 已下载数据

| 数据集/模型 | 来源 | 路径 | 大小 | 备注 |
|---|---|---|---|---|
| TerminalHorizon-Environment | HuggingFace | `/storage/openpsi/users/chen/code_data/datasets/TerminalHorizon-Environment` | 6.44 GiB | 8 个分卷 tar.gz + metadata，sha256 全部校验通过 |
