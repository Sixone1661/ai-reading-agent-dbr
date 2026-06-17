# ai-reading-agent-dbr

面向研究生批判性文献阅读能力提升的 AI 苏格拉底式对话支架 V1.0。本项目是一个本地运行的 Streamlit + SQLite 教育技术研究原型，用于 15 周研究生课程中的若干文献阅读节点，支持保存对话日志和批判性阅读笔记，便于后续设计研究（DBR）数据分析。

## 功能

- 学生端
  - 输入学生编号、周次节点（T0/T1/T2/T3）、文献题目、任务类型，并可手动粘贴文献文本或上传 PDF / Word 文献。
  - 与 AI 进行苏格拉底式对话。
  - AI 以追问方式引导学生分析研究问题、理论框架、研究方法、证据链、局限贡献和迁移反思。
  - 保存每轮学生回答和 AI 提问。
  - 填写并保存批判性阅读笔记模板。

- 教师端
  - 查看所有学生阅读笔记。
  - 查看每个学生的对话日志。
  - 导出对话日志 CSV。
  - 导出阅读笔记 CSV。

- 数据库
  - SQLite 数据库文件：`reading_agent.db`。
  - 数据表：`sessions`、`messages`、`reading_notes`。

## 文件结构

```text
ai-reading-agent-dbr/
├── app.py
├── database.py
├── export_utils.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── prompts/
    └── system_prompt.md
```

## 安装与运行

### 1. 进入项目目录

```powershell
cd "D:\ALL_Files\University_all_files\MyPKU\pku2025-2028\pku2026\2026学习类\1.2026春课程们\26春9教育技术研究方法\5.作业\基于设计的研究\期末作业\ai-reading-agent-dbr"
```

### 2. 创建虚拟环境

由于本项目目录路径较深，Windows 在安装 Streamlit 时可能出现 `WinError 206 文件名或扩展名太长`。推荐把虚拟环境放在短路径，例如：

```powershell
python -m venv D:\environment
D:\environment\Scripts\Activate.ps1
```

如果你希望把虚拟环境放在项目内，也可以尝试：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

若 PowerShell 阻止激活脚本，可临时执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. 安装依赖

激活虚拟环境后，在项目目录运行：

```powershell
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env`：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，填入你的 OpenAI API Key：

```text
OPENAI_API_KEY=你的真实 API Key
OPENAI_MODEL=gpt-4.1-mini
```

不要把 `.env` 上传到公开仓库。

### 5. 启动应用

```powershell
streamlit run app.py
```

启动后浏览器会打开本地地址，通常是：

```text
http://localhost:8501
```

## 使用建议

1. 学生先在左侧创建阅读会话。
2. 在“AI 对话”中根据 AI 的追问逐步分析文献。
3. 在“批判性阅读笔记”中整理阶段性理解。
4. 教师在“教师端”查看数据并导出 CSV。

## 注意

- 本版本不包含复杂登录系统，适合作为本地课堂研究原型。
- 如果没有配置 `OPENAI_API_KEY`，页面会显示友好提示，笔记保存仍可使用，但 AI 对话不可用。
- 数据默认保存在项目目录下的 `reading_agent.db`。


