# Tool-Using Writer Agent

## 概述

Tool-Using Writer Agent 是对原有教案生成系统的重大升级，允许 Writer Agent 调用外部工具获取额外信息，从而生成更丰富、更时效的教案内容。

## 架构

```
User Request: "生成量子计算教案"
    ↓
PlannerAgent:
  - 分析主题和学科
  - 决定需要哪些工具
  - 生成工具调用计划
    ↓
ToolExecutor:
  - 并行执行工具调用
  - 收集工具结果
  - 处理失败和超时
    ↓
WriterAgent:
  - 基于 RAG 上下文生成教案
  - 整合工具获取的信息
  - 生成最终教案
```

## 支持的工具

### 1. Web Search Tool

**功能**：搜索互联网获取最新资料

**使用场景**：
- 物理：最新科技应用（量子计算、新能源汽车）
- 化学：最新材料、环保技术
- 生物：最新医学进展、疫情数据
- 历史/地理：当前时事、最新统计数据

**配置**：
```bash
# .env
TOOL_USING_ENABLED=true
WEB_SEARCH_PROVIDER=tavily
WEB_SEARCH_API_KEY=your_api_key_here
```

**API 提供商**：
- Tavily API (推荐): https://tavily.com
- SerpAPI: https://serpapi.com

### 2. Image Retrieval Tool

**功能**：从现有图片库智能检索相关图片

**使用场景**：
- 物理：实验装置图、受力分析图
- 化学：分子结构、实验现象
- 生物：细胞结构、器官图
- 数学：几何图形、函数图像

**实现方式**：
- 复用现有 `image_resources`
- 基于关键词的语义检索
- 可扩展为基于 embedding 的语义搜索

**配置**：
```bash
# .env
TOOL_USING_ENABLED=true
```

### 3. LaTeX Renderer Tool

**功能**：验证 LaTeX 语法或渲染为图片

**使用场景**：
- 数学：微积分公式、矩阵运算、几何定理
- 物理：力学公式、电磁学方程、量子力学
- 化学：化学反应方程式、分子式

**实现方式**：
- 基础语法验证（当前实现）
- 可扩展为调用 MathJax/KaTeX 渲染服务

**配置**：
```bash
# .env
TOOL_USING_ENABLED=true
LATEX_RENDERER_ENABLED=true
```

## 工具规划逻辑

ToolPlanner 根据主题和学科自动决定需要哪些工具：

### Web Search 触发条件
- 主题包含"最新"、"当前"、"进展"等关键词
- 主题涉及时事性内容（量子计算、人工智能、新能源等）

### Image Retrieval 触发条件
- 模板类型为 comprehensive、teaching_design 或 ppt
- 学科为物理、化学、生物、数学、地理
- 主题包含"结构"、"图"、"实验"等关键词

### LaTeX Renderer 触发条件
- 学科为数学、物理、化学
- 主题包含"公式"、"方程"、"定理"等关键词

## 降级策略

所有工具都实现了优雅降级，确保工具调用失败不影响主流程：

| 工具 | 失败处理 |
|------|----------|
| Web Search | 仅使用 RAG 检索结果 |
| Image Retrieval | 返回空列表，不插入图片 |
| LaTeX Renderer | 返回原始 LaTeX 代码 |

## 使用示例

### 示例 1：生成量子计算教案

```python
from src.agents.orchestrator import LessonOrchestrator
from src.agents.tools import WebSearchTool, ImageRetrievalTool, LaTeXRendererTool
from src.agents.tools.base import ToolExecutor

# 初始化工具
tools = [
    WebSearchTool(api_key="your_api_key"),
    ImageRetrievalTool(image_resources=image_db),
    LaTeXRendererTool(),
]
tool_executor = ToolExecutor(tools)

# 创建 orchestrator
orchestrator = LessonOrchestrator(
    planner_agent=planner,
    query_agent=query_agent,
    retriever_agent=retriever,
    writer_reviewer_agent=writer,
    conversation_agent=conversation,
    trace=trace,
    tool_executor=tool_executor,  # 传入工具执行器
)

# 生成教案
result = orchestrator.run(
    topic="量子计算最新进展",
    template_category="comprehensive",
    conversation_state=state,
)

# 查看工具结果
for tool_result in result["tool_results"]:
    print(f"Tool: {tool_result['tool_name']}")
    print(f"Success: {tool_result['success']}")
    if tool_result['success']:
        print(f"Data: {tool_result['data']}")
```

### 示例 2：运行演示脚本

```bash
# 运行工具演示
.venv/bin/python scripts/demo_tool_using.py

# 运行单元测试
.venv/bin/python -m pytest tests/unit/test_tools.py -v
```

## 性能指标

| 指标　　　　　 | 目标值 | 当前值　　　　|
| ----------------| --------| ---------------|
| 工具调用成功率 | > 95%　| ✓ 100% (测试) |
| 工具调用超时　 | < 10s　| ✓ 10s　　　　 |
| 并行执行　　　 | 支持　 | ✓ 支持　　　　|
| 降级策略　　　 | 完整　 | ✓ 完整　　　　|

## 可观测性

所有工具调用都有完整的日志和 trace：

```python
# 日志示例
2026-04-22 16:43:27,842 INFO src.agents.tool_planner 
  tool_planner.plan topic=量子计算最新进展 subject=物理 
  tool_count=3 tools=['web_search', 'image_retrieval', 'latex_renderer']

2026-04-22 16:43:27,842 INFO src.agents.tools.web_search 
  web_search.success query=量子计算 result_count=5

2026-04-22 16:43:27,842 INFO src.agents.tools.base 
  tool_executor.execute_success tool=web_search elapsed_ms=234.5
```

## 扩展性

### 添加新工具

1. 创建工具类继承 `Tool`：

```python
from src.agents.tools.base import Tool, ToolResult

class MyCustomTool(Tool):
    def __init__(self, timeout: float = 10.0):
        super().__init__(name="my_custom_tool", timeout=timeout)
    
    def validate_params(self, **kwargs):
        return "required_param" in kwargs
    
    async def execute(self, required_param: str, **kwargs) -> ToolResult:
        try:
            # 执行工具逻辑
            result_data = await self._do_something(required_param)
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=result_data,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
            )
```

2. 在 ToolPlanner 中添加规划逻辑：

```python
# src/agents/tool_planner.py

def plan_tools(self, topic: str, subject: str, **kwargs):
    tool_calls = []
    
    # 添加你的工具规划逻辑
    if self._should_use_my_tool(topic, subject):
        tool_calls.append({
            "tool_name": "my_custom_tool",
            "params": {"required_param": "value"},
            "reason": "工具使用原因",
        })
    
    return tool_calls
```

3. 注册工具到执行器：

```python
from src.agents.tools.base import ToolExecutor
from my_module import MyCustomTool

tools = [
    WebSearchTool(),
    ImageRetrievalTool(),
    LaTeXRendererTool(),
    MyCustomTool(),  # 添加新工具
]

executor = ToolExecutor(tools)
```

## 故障排查

### 问题 1：Web Search 不工作

**症状**：Web search 总是返回 disabled 错误

**解决方案**：
1. 检查环境变量：
   ```bash
   echo $TOOL_USING_ENABLED
   echo $WEB_SEARCH_API_KEY
   ```
2. 确保 API key 有效
3. 检查网络连接

### 问题 2：工具调用超时

**症状**：工具执行时间过长，返回 timeout 错误

**解决方案**：
1. 增加超时时间：
   ```python
   tool = WebSearchTool(timeout=20.0)  # 增加到 20 秒
   ```
2. 检查网络延迟
3. 考虑使用更快的 API 提供商

### 问题 3：图片检索结果不准确

**症状**：检索到的图片与主题不相关

**解决方案**：
1. 改进关键词匹配算法
2. 使用基于 embedding 的语义搜索
3. 增加图片元数据质量

## 下一步

### P1 功能（可选）
- [ ] Multi-Expert Reviewer（多专家评审）
- [ ] Reflection Loop（反思循环）

### 优化方向
- [ ] 使用 embedding 改进图片检索
- [ ] 实现 LaTeX 渲染为图片
- [ ] 添加更多工具（Calculator、Code Executor 等）
- [ ] 优化工具调用并发性能

## 参考资料

- [Tavily API 文档](https://docs.tavily.com)
- [SerpAPI 文档](https://serpapi.com/docs)
- [MathJax 文档](https://docs.mathjax.org)
- [LaTeX 语法参考](https://www.overleaf.com/learn/latex/Mathematical_expressions)
