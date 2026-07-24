# Relight 图片重打光流水线

本项目只处理已有图片目录：Qwen VL判断图片是否适合重打光并生成中英文指令，通过后调用ToAPIs图片模型生成一张重打光结果。它不负责下载数据集，也不执行尺寸、水印、人脸或场景筛选。

## 安装

```powershell
git clone https://github.com/WolfRanok/relight-pipeline.git
cd relight-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.venv`是每台机器独立创建的运行环境，按Python项目惯例不会上传GitHub；
第三方依赖由`requirements.txt`或`pyproject.toml`声明并自动安装。其他项目也可
直接安装本仓库的核心功能：

```powershell
pip install -e .
```

需要OSS功能时安装可选依赖：

```powershell
pip install -e ".[oss]"
```

需要运行测试时安装开发依赖：

```powershell
pip install -e ".[dev,oss]"
```

Qwen密钥保存在`APIKEY.txt`，ToAPIs密钥保存在`TOAPIS_APIKEY.txt`，每个文件只放一行Key。也可以使用`DASHSCOPE_API_KEY`和`TOAPIS_API_KEY`环境变量临时覆盖。密钥不会进入运行结果。

## 使用

```powershell
# 审核并处理目录中的全部图片
python relight_demo.py --input-dir ".\data\Human\images"

# 最多消耗10个成功或失败名额；VL主动skip不占名额
python relight_demo.py --input-dir ".\data\Human\images" --count 10

# 自定义输出数据集名称
python relight_demo.py --input-dir ".\data\Human\images" --name MyDataset

# 续跑同一次运行
python relight_demo.py --resume ".\output\2026-07-23\Human_1"
```

程序递归扫描JPG、JPEG、PNG、WEBP、TIF、TIFF和BMP。输入目录名为`images`时，默认使用父目录名作为输出名称。

## 输出

```text
output/YYYY-MM-DD/数据集名称_编号/
├─ 图片/原图片相对路径（含文件名）/
│  ├─ original.<原扩展名>
│  └─ relight.<原扩展名>
├─ 提示词/对应相对路径.json
└─ .pipeline/
   ├─ state.sqlite3
   ├─ results.jsonl
   ├─ skipped.jsonl
   ├─ failed.jsonl
   ├─ summary.json
   └─ run_config.json
```

提示词JSON只包含`edit_prompt`中文指令和`edit_prompt_en`英文原指令。`RELIGHT_RESOLUTION="2k"`控制服务端输出档位，不是输入筛选条件；程序会将源图比例传给模型。

## OSS选项

`env.py`中的`RELIGHT_OSS_ENABLED`默认是`False`。改为`True`后，源图按SHA-256缓存到私有OSS，生图服务使用短期签名URL，成功交付物同时同步到OSS和本地。OSS参数默认读取项目根目录下被Git忽略的`OSS_CONFIG.local.md`，环境变量优先。

阿里云OSS官方SDK是可选依赖：OSS关闭时不会加载；OSS开启但未安装SDK时，
程序会提示执行`pip install -e ".[oss]"`，不会再因模块顶层导入而直接启动失败。

## 调度和错误恢复

每张图片通过VL后立即开始生图，不等待一批图片全部审核或生成完毕。各阶段并发由`env.py`分别控制。模型无渠道、鉴权或账户硬错误会触发熔断，终端会给出`--resume`命令；已有任务ID不会重复提交。

## 测试

```powershell
python -m pytest -q
```

测试全部使用模拟客户端，不会调用真实模型。
