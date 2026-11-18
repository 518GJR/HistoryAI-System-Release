# HistoryAI-System-Release

毕业设计“面向中小学历史知识点的语音驱动问答与知识梳理系统”的独立公开展示版本。代码来自HistoryAI-System最终开发工作区，发布目录不继承原Git历史。

## 功能与结构

Vue3/Vite 前端 → FastAPI → 关键词或本地BGE向量检索 → 资料支持判断与AI回答 → SQLAlchemy历史记录；支持知识梳理、ASR转写、TTS播放。

- backend/app/api：问答、历史、知识、语音接口。
- backend/app/services：检索、模型、语音、证据判断与评估服务。
- backend/scripts：三种模式实验、自动评分、恢复、裁定与统计入口。
- backend/tests：原有自动化测试保留；部分需要不公开的本地数据。
- frontend/src：Vue页面；docs/evaluation：公开题集、签核、评分和已有统计图。
- docs/evaluation/public_scores：最终人工裁定1446条及100题数值汇总，分数未改变。

请求mode支持none、keyword、vector；none调用模型，其他模式检索后判断资料支持度。模型不可用时按现有路径报错或降级，不能把模板回答当作真实模型生成。

## 安装与演示

Python依赖以backend/requirements.txt为准；Node建议18以上。首次安装依赖需要网络，离线验证阶段不安装也不下载模型。

从仓库根目录执行PowerShell：

```powershell
Copy-Item .env.example .env
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

编辑根目录.env，演示可用 `DATABASE_URL=sqlite:///./history_ai.db`，`KNOWLEDGE_MODE=demo`，`RETRIEVAL_MODE=keyword`。AI/ASR/TTS默认disabled；真实服务需自行填写对应PROVIDER、MODEL、BASE_URL、API_KEY，凭据只保存在本地。demo数据仅用于基础流程演示，不代表正式教材覆盖或实验语料。

```powershell
# backend目录
.\.venv\Scripts\python.exe -m app.init_db
.\.venv\Scripts\python.exe -m app.scripts.import_history_knowledge
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

另开终端：

```powershell
cd frontend
npm ci
npm run dev
```

访问 http://localhost:5173 ，接口文档 http://localhost:8000/docs 。录音需要麦克风权限及安全上下文；disabled配置不提供真实语音或none模型回答。数据库不可用时历史可能保存失败。

## 合法本地教材配置

本公开目录不提供教材PDF、扫描件、整章内容或教材原文知识库。用户须自行准备有权使用的数据。

基础教材模式固定读取 `data/textbook_history_knowledge.json`，schema校验见 `backend/app/services/rag_service.py:load_textbook_knowledge_from_json`：根对象包含book、unit、lessons、sources、knowledge_points，来源ID、知识点ID及课次关联必须一致。仅元数据不能替代资料支持判断要求的真实本地证据。准备文件后设置KNOWLEDGE_MODE=textbook，缺失或错误数据会明确报错。

签核扩展库需显式配置SIGNED_TEXTBOOK_DATABASE_URL，并使用 `python -m app.scripts.import_signed_textbook_knowledge --help` 指定本地知识与来源文件；默认staging文件未公开。不得将连接到正式数据库的导入命令当作无副作用检查。

向量模式还需本地模型及与语料匹配的索引。按.env.example的模型/revision配置，backend目录执行 `python -m app.scripts.build_textbook_vector_index`；首次模型下载需用户自行安排，已有缓存可开启HF_HUB_OFFLINE=1和TRANSFORMERS_OFFLINE=1。签核数据库启用时必须指定新的--output。

## 测试与实验

```powershell
cd backend
$env:DATABASE_URL='sqlite:///:memory:'
$env:AI_PROVIDER='disabled'
$env:ASR_PROVIDER='disabled'
$env:TTS_PROVIDER='disabled'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe -m pytest tests -q
```

完整测试包含本地教材、原始评分和签核数据依赖，缺少时会失败；不会隐藏或删除这些测试。公开目录实际验证见PUBLIC_RELEASE_MANIFEST.md。离线可独立运行的子集见该清单，不等于完整教材验收。

正式100题公开副本在docs/evaluation/releases/formal_100_v1.0，保留原题、评分规则及candidate状态；教材摘录被明确标记为本地保留。实际实验需准备合法语料和模型，从backend执行：

```powershell
python scripts/run_evaluation.py --rubric ../docs/evaluation/releases/formal_100_v1.0/formal_evaluation_100.json --expected-questions 100 --modes none keyword vector --output-root ../tmp/evaluation
```

该命令会调用配置的模型服务，并非离线测试；--skip-scoring只关闭自动评分。固定模型、题集、Prompt、语料、评分规则后才可解释三模式差异。

人工公开分数来自最终审核表，不以旧自动重评分pending替代当前人工结论。原始长回答、教材证据和部分重评分明细因公开限制不分发，因此公开版不承诺逐字重放全部旧实验。六张统计图来自既有成果，未重跑实验或改变分数。

真实云服务、MySQL和浏览器麦克风未在本轮复测。公开数据不是新的教材授权，第三方依赖遵循各自许可。未擅自给原项目增设开源许可。
