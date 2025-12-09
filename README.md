<h1 align="center">智能 RAG MCP 服务器</h1>

<p align="center">
  <em>为 AI 助手提供网页抓取和 RAG 检索能力</em>
</p>

基于 [Model Context Protocol (MCP)](https://modelcontextprotocol.io) 实现的智能服务器，集成了 [Jina](https://jina.ai) 和 [Supabase](https://supabase.com/)，为 AI 助手提供强大的网页抓取和 RAG 检索能力。

使用这个 MCP 服务器，你可以**抓取任何内容**，然后**在任何地方使用这些知识**进行 RAG 检索。

## 概述

本服务器提供了一套工具，让 AI 助手能够抓取网站内容、将内容存储到向量数据库（Supabase）中，并对抓取的内容执行 RAG 检索。

服务器包含多种高级 RAG 策略，可以按需启用以提升检索质量：
- **上下文嵌入**：增强语义理解
- **混合搜索**：结合向量搜索和关键词搜索
- **智能代码提取**：专门提取和检索代码示例
- **重排序**：使用交叉编码器模型提升结果相关性
- **知识图谱**：检测 AI 幻觉并分析代码仓库

详细配置方法请参考下方的[配置章节](#配置)。

## 核心功能

- **智能 URL 识别**：自动检测和处理不同类型的 URL（网页、站点地图、文本文件）
- **递归抓取**：跟随内部链接发现内容
- **并行处理**：高效地同时抓取多个页面
- **智能分块**：按标题和大小智能分割内容
- **向量搜索**：对抓取的内容执行 RAG 检索，可按数据源过滤
- **源管理**：检索可用的数据源以指导 RAG 过程

## 工具列表

### 核心工具（始终可用）

1. **`crawl_single_page`**：快速抓取单个网页并存储到向量数据库
2. **`smart_crawl_url`**：智能抓取整个网站（支持站点地图、llms-full.txt 或递归抓取普通网页）
3. **`get_available_sources`**：获取数据库中所有可用的数据源（域名）列表
4. **`perform_rag_query`**：使用语义搜索查找相关内容，可选择按源过滤
5. **`ingest_local_files`**：直接读取本地 Markdown/文本/PDF 文件，分块后推送到 Supabase

### 条件工具

6. **`search_code_examples`**（需要 `USE_AGENTIC_RAG=true`）：专门搜索代码示例及其摘要

### 知识图谱工具（需要 `USE_KNOWLEDGE_GRAPH=true`）

7. **`parse_github_repository`**：将 GitHub 仓库解析为 Neo4j 知识图谱
8. **`check_ai_script_hallucinations`**：分析 Python 脚本，检测 AI 幻觉
9. **`query_knowledge_graph`**：探索和查询 Neo4j 知识图谱

### 本地文件导入示例

```json
{
  "tool": "ingest_local_files",
  "file_paths": "[\"docs/intro.md\", \"notes/**/*.md\", \"spec.pdf\"]",
  "recursive": true
}
```

## 前置要求

- [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/)（推荐使用容器运行）
- [Python 3.12+](https://www.python.org/downloads/)（直接运行需要）
- [Supabase](https://supabase.com/) 账号（用于 RAG 数据库）
- [Jina AI API key](https://jina.ai/embeddings/)（用于嵌入和重排序）
- 任何兼容 OpenAI 的 LLM 端点（如 OpenRouter、Groq、Together）
- [Neo4j](https://neo4j.com/)（可选，用于知识图谱功能）

## 安装

### 方式一：使用 Docker（推荐）

1. 克隆仓库：
   ```bash
   git clone <your-repo-url>
   cd mcp-rag
   ```

2. 构建 Docker 镜像：
   ```bash
   docker build -t mcp/rag-server --build-arg PORT=8051 .
   ```

3. 根据下方配置章节创建 `.env` 文件

### 方式二：直接使用 uv

1. 克隆仓库：
   ```bash
   git clone <your-repo-url>
   cd mcp-rag
   ```

2. 安装 uv：
   ```bash
   pip install uv
   ```

3. 创建并激活虚拟环境：
   ```bash
   uv venv
   # Windows: .venv\Scripts\activate
   # Mac/Linux: source .venv/bin/activate
   ```

4. 安装依赖：
   ```bash
   uv pip install -e .
   ```

5. 创建 `.env` 文件
   - 设置 `TENANT_ID` 为唯一值（如组织名称）以隔离 Supabase 数据
   - 客户端可在每次 MCP 请求中携带 `X-Tenant-ID` header 来动态指定租户

## 数据库设置

运行服务器前，需要在 Supabase 中设置数据库：

1. 在 Supabase 控制台打开 SQL 编辑器（如需要先创建新项目）

2. 创建新查询并粘贴 `crawled_pages.sql` 的内容

3. 运行查询以创建必要的表和函数

### 嵌入缓存与增量更新

默认架构包含：

- `crawled_pages` 和 `code_examples` 表中的 `content_hash`、`embedding_model`、`embedding_cached_at` 字段
- 专用的 `embedding_cache` 表，按 `(tenant_id, content_hash, model_name)` 键索引

抓取/导入工具会：

1. 在嵌入前对每个块进行哈希
2. 在 `embedding_cache` 中查找可复用的嵌入，命中缓存则跳过 API 调用
3. 将新嵌入插入缓存，按租户和嵌入版本/模型隔离
4. 标记过期缓存行（`refreshed_at` 超过 `EMBEDDING_CACHE_TTL_SECONDS`）以便异步刷新

通过环境变量配置缓存行为：

```
EMBEDDING_VERSION=                  # 可选的嵌入配置名称（默认为 EMBEDDING_MODEL）
EMBEDDING_CACHE_ENABLED=true
EMBEDDING_CACHE_TTL_SECONDS=604800  # 7 天
```

升级嵌入模型时，更新 `EMBEDDING_VERSION` 以强制缓存失效并增量重新填充。

### 智能抓取配置

`smart_crawl_url` 使用 Jina 的 “Links/Buttons” 摘要来发现可遍历的内部链接。可通过环境变量控制抓取范围：

```
SMART_CRAWL_MAX_PAGES=30            # 单次深度抓取的最大页面数
SMART_CRAWL_MAX_LINKS_PER_PAGE=20   # 每个页面最多提取的链接数
```

抓取流程使用 BFS 队列（按域名去重）。Jina 代理失败时自动回退到直接请求，但此时无法获得链接列表，因此只会抓取起始页面。

## 知识图谱设置（可选）

要启用 AI 幻觉检测和仓库分析功能，需要设置 Neo4j。

**注意**：知识图谱功能目前与 Docker 不完全兼容，建议直接通过 uv 运行。

### 安装 Neo4j

**方式一：使用 Neo4j Desktop**

1. 从 [neo4j.com/download](https://neo4j.com/download/) 下载 Neo4j Desktop

2. 创建新数据库：
   - 打开 Neo4j Desktop
   - 创建新项目和数据库
   - 为 `neo4j` 用户设置密码
   - 启动数据库

3. 记录连接信息：
   - URI: `bolt://localhost:7687`（默认）
   - 用户名: `neo4j`（默认）
   - 密码: 创建时设置的密码

**方式二：使用 Docker**

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest
```

## 配置

在项目根目录创建 `.env` 文件，包含以下变量：

```bash
# MCP 服务器配置
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse
TENANT_ID=default
TENANT_HEADER_NAME=X-Tenant-ID

# Jina 嵌入配置
JINA_API_KEY=your_jina_api_key
JINA_API_URL=https://api.jina.ai/v1/embeddings
EMBEDDING_MODEL=jina-embeddings-v3
JINA_EMBEDDING_TASK=text-matching
EMBEDDING_DIM=1024

# Jina 重排序配置（可选）
JINA_RERANK_URL=https://api.jina.ai/v1/rerank
JINA_RERANK_MODEL=jina-reranker-v3

# LLM 配置（用于摘要/上下文嵌入）
MODEL_CHOICE=openrouter/anthropic/claude-3.5-haiku
LLM_API_KEY=your_llm_key
LLM_API_BASE=https://openrouter.ai/api/v1

# RAG 策略开关（设置为 "true" 或 "false"，默认为 "false"）
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false

# Supabase 配置
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key

# Neo4j 配置（知识图谱功能需要）
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### RAG 策略说明

服务器支持五种 RAG 策略，可独立启用：

#### 1. USE_CONTEXTUAL_EMBEDDINGS（上下文嵌入）

为每个文本块添加整个文档的上下文信息，提升检索精度。

- **适用场景**：技术文档中术语在不同章节有不同含义时
- **代价**：索引速度较慢，需额外 LLM API 调用

#### 2. USE_HYBRID_SEARCH（混合搜索）

结合向量搜索和关键词搜索，提供更全面的结果。

- **适用场景**：需要精确匹配技术术语、函数名时
- **代价**：查询稍慢，但无额外 API 成本

#### 3. USE_AGENTIC_RAG（智能代码提取）

专门提取和存储代码示例（≥300 字符），生成摘要并单独索引。

- **适用场景**：AI 编程助手需要查找代码示例和实现模式时
- **代价**：抓取速度显著变慢，需额外 LLM API 调用

#### 4. USE_RERANKING（重排序）

使用 Jina 重排序 API 对初始检索结果重新评分。

- **适用场景**：搜索精度至关重要时
- **代价**：每次查询增加一次 API 调用（~100-200ms）

#### 5. USE_KNOWLEDGE_GRAPH（知识图谱）

使用 Neo4j 知识图谱检测 AI 幻觉并分析代码仓库。

- **适用场景**：需要验证 AI 生成的代码是否符合真实仓库结构时
- **代价**：需要 Neo4j 基础设施，大型代码库解析较慢
- **注意**：目前与 Docker 不完全兼容，建议通过 uv 运行

### 推荐配置

**通用文档 RAG：**

```bash
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=true
```

**AI 编程助手（含代码示例）：**

```bash
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
USE_KNOWLEDGE_GRAPH=false
```

**AI 编程助手（含幻觉检测）：**

```bash
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
USE_KNOWLEDGE_GRAPH=true
```

**快速基础 RAG：**

```bash
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false
```

## 运行服务器

### 使用 Docker

```bash
docker run --env-file .env -p 8051:8051 mcp/rag-server
```

### 使用 Python

```bash
uv run src/rag_mcp.py
```

服务器将在配置的主机和端口上启动。

## 集成到 MCP 客户端

### SSE 配置

服务器运行后，可使用以下配置连接：

```json
{
  "mcpServers": {
    "rag-server": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

**注意事项：**

- **Windsurf 用户**：使用 `serverUrl` 而非 `url`
- **Docker 用户**：如果客户端在不同容器中运行，使用 `host.docker.internal` 代替 `localhost`
- **Claude Code 用户**：

```bash
claude mcp add-json rag-server '{"type":"http","url":"http://localhost:8051/sse"}' --scope user
```

### Stdio 配置

在 Claude Desktop、Windsurf 或其他 MCP 客户端中添加配置：

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["path/to/src/rag_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "JINA_API_KEY": "your_jina_api_key",
        "LLM_API_KEY": "your_llm_api_key",
        "LLM_API_BASE": "https://openrouter.ai/api/v1",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

### Docker Stdio 配置

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "--env-file", ".env", "mcp/rag-server"],
      "env": {
        "TRANSPORT": "stdio"
      }
    }
  }
}
```

## 知识图谱架构

知识图谱系统将代码仓库结构存储在 Neo4j 中：

### 核心组件（`knowledge_graphs/` 目录）

- **`parse_repo_into_neo4j.py`**：克隆并分析 GitHub 仓库，提取 Python 类、方法、函数和导入关系
- **`ai_script_analyzer.py`**：使用 AST 解析 Python 脚本，提取导入、类实例化、方法调用等
- **`knowledge_graph_validator.py`**：验证 AI 生成的代码，检测幻觉（不存在的方法、错误参数等）
- **`hallucination_reporter.py`**：生成幻觉检测报告，包含置信度分数和建议
- **`query_knowledge_graph.py`**：交互式 CLI 工具（功能已集成到 MCP 工具中）

### 知识图谱架构

**节点类型：**

- `Repository`：GitHub 仓库
- `File`：Python 文件
- `Class`：Python 类
- `Method`：类方法
- `Function`：独立函数
- `Attribute`：类属性

**关系类型：**

- `Repository` -[:CONTAINS]-> `File`
- `File` -[:DEFINES]-> `Class`
- `File` -[:DEFINES]-> `Function`
- `Class` -[:HAS_METHOD]-> `Method`
- `Class` -[:HAS_ATTRIBUTE]-> `Attribute`

### 使用流程

1. **解析仓库**：使用 `parse_github_repository` 工具克隆并分析开源仓库
2. **验证代码**：使用 `check_ai_script_hallucinations` 工具验证 AI 生成的 Python 脚本
3. **探索知识**：使用 `query_knowledge_graph` 工具探索可用的仓库、类和方法

## 自定义开发

本实现为构建更复杂的 MCP 服务器提供了基础：

1. 使用 `@mcp.tool()` 装饰器添加自定义工具
2. 创建自定义 lifespan 函数添加依赖
3. 修改 `utils.py` 添加辅助函数
4. 扩展抓取能力，添加更多专用爬虫
