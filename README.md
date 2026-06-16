# Document AI Pipeline

本项目是一个本地单据识别演示系统，用于客户试用和现场演示：

- 上传发货单、出货单、验收单等图片
- 输入客户自定义提取要求
- 调用本地 vLLM 多模态模型
- 展示 JSON、全部字段、表格明细和生产接口视图
- 支持图片放大、缩小、旋转后再提交识别
- 输出模型 JSON，并后台更新 Excel

## 启动方式

推荐使用项目目录下的 uv / `.venv` 环境：

```powershell
cd "C:\Users\zeng\Documents\New project 2\document_ai_pipeline"
.\.venv\Scripts\python.exe -m waitress --host=0.0.0.0 --port=7861 demo_app.app:app
```

本机访问：

```text
http://127.0.0.1:7861/
```

局域网客户演示时访问：

```text
http://服务电脑IP:7861/
```

## 模型配置

默认调用本地 vLLM OpenAI-compatible 服务：

```yaml
model:
  provider: openai_compatible
  base_url: http://192.168.2.85:8000/v1
  model_name: Qwen3.6-35B-A3B-FP8
```

配置文件：`config.yaml`

## 识别模式

页面提供两种模式：

- 全量识别：尽量抽取图片中所有有价值字段和表格
- 定向识别：把用户提示词当成字段白名单，只提取用户要求的数据

示例定向提示词：

```text
只提取合同编号、SN、签字日期。其他字段不要输出。
```

## 输出目录

- `data/input/`：上传图片保留位置
- `data/output/model_json/`：模型输出 JSON
- `data/output/tables/model_results.xlsx`：后台更新的 Excel 汇总
- `data/output/debug/`：任务调试信息
- `data/logs/`：日志和状态库

这些运行数据默认被 `.gitignore` 排除，不会上传到 GitHub。

## GitHub 私有仓库

本项目适合上传成私有仓库。注意不要提交：

- `.venv/`
- `data/input/` 里的客户图片
- `data/output/` 里的识别结果
- `data/logs/` 里的日志和数据库

`.gitignore` 已经配置好这些规则。
