<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.agents.default

## AgentSession

```python
@dataclass
class AgentSession
```

Mutable per-session state for the default LLM agent.

### 参数

- `messages` (`list[BaseMessage]`)
  Conversation history, including system, user, assistant, and tool
  messages.
- `metadata` (`dict[str, Any]`)
  Session-scoped mutable state used by context updates and dynamic tools.

### 类字段

- `messages: list[BaseMessage]` = `field(default_factory=list)`
- `metadata: dict[str, Any]` = `field(default_factory=dict)`

## get_context_data

```python
def get_context_data(session: AgentSession, context_type: str) -> dict[str, Any]
```

Return stored agent-context data for one logical context type.

### 参数

- `session` (`AgentSession`)
  Mutable runtime session state.
- `context_type` (`str`)
  Logical context stream name such as ``"caption"``.

### 返回

- `dict[str, Any]`
  Stored payload for the context type, or an empty dict.

## MutableToolProvider

```python
class MutableToolProvider
```

Provide a clone-safe list of tools that can be extended at runtime.

### 参数

- `tools` (`list[BaseTool | Callable[[], BaseTool]] | None, optional`)
  Initial tool instances or factories.

### 方法

#### __init__

```python
def __init__(self, tools: Optional[list[BaseTool | Callable[[], BaseTool]]] = None) -> None
```

#### add_tools

```python
def add_tools(self, tools: list[BaseTool | Callable[[], BaseTool]]) -> None
```

Append tools or tool factories.

##### 参数

- `tools` (`list[BaseTool | Callable[[], BaseTool]]`)
  Tool instances or factories to append.

#### get_tool_specs

```python
def get_tool_specs(self) -> list[Callable[[], BaseTool]]
```

Return clone-safe tool factories.

##### 返回

- `list[Callable[[], BaseTool]]`
  Normalized factories used by this provider.

#### get_tools

```python
def get_tools(self, session: AgentSession) -> list[BaseTool]
```

Return session-scoped tool instances for the current session.

##### 参数

- `session` (`AgentSession`)
  Runtime session state used to cache tool instances.

##### 返回

- `list[BaseTool]`
  Tool instances reused within the current session.

#### normalize_tool_specs

```python
def normalize_tool_specs(tools: list[BaseTool | Callable[[], BaseTool]]) -> list[Callable[[], BaseTool]]
```

Normalize tools into zero-argument factories.

##### 参数

- `tools` (`list[BaseTool | Callable[[], BaseTool]]`)
  Tool instances or factories.

##### 返回

- `list[Callable[[], BaseTool]]`
  Factory list.

## DefaultToolProvider

```python
class DefaultToolProvider(MutableToolProvider)
```

Build the default tool set for each session.

### 方法

#### __init__

```python
def __init__(self, *, voice_names: Optional[list[str]] = None, emotions: Optional[list[str]] = None, tools: Optional[list[BaseTool | Callable[[], BaseTool]]] = None) -> None
```

Initialize the default tool provider.

##### 参数

- `voice_names` (`list[str] | None, optional`)
  Available voice names for the voice-switch tool.
- `emotions` (`list[str] | None, optional`)
  Available emotions for the emotion tool.
- `tools` (`list[BaseTool | Callable[[], BaseTool]] | None, optional`)
  Explicit tool instances or factories. When omitted, the default
  tool set is used.

#### get_tools

```python
def get_tools(self, session: AgentSession) -> list[BaseTool]
```

Build the tool set for the current session.

##### 参数

- `session` (`AgentSession`)
  Runtime session state.

##### 返回

- `list[BaseTool]`
  Enabled tools for this session.

## summarize_embedding_doc

```python
def summarize_embedding_doc(doc: str, max_sentences: int | None = None) -> str
```

Build a lightweight extractive summary for uploaded text.

### 参数

- `doc` (`str`)
  Document text.
- `max_sentences` (`int | None, optional`)
  Maximum number of summary sentences.

### 返回

- `str`
  Extractive summary.

## DefaultAgent

```python
@model
class DefaultAgent(Agent)
```

Default speech-first conversational agent implementation.

### 类字段

- `BASE_PROMPT: str` = `'\nYou are a friendly conversational partner whose response will be converted to speech using TTS. Please follow rules below:\n1. Respond with the same language as user.\nExamples:\n- user: 你好。\n- assistant: 你好呀，今天感觉怎么样？\n- user: Hello.\n- assistant: Hello, how are you today?\n\n2. Your response should not contain content that cannot be synthesize by the TTS model, such as parentheses, ordered lists (starting by - ), etc. Numbers should be written in English words rather than Arabic numerals.\n\n3. Your response should be informative and adequately detailed, but avoid unnecessary repetition or filler. Keep it suitable for spoken delivery.\n\n4. If you find user input (ASR result) unclear, incomplete, or likely incorrect — for example:\n- contains obvious ASR hallucinations,\n- contains broken words or meaningless fragments,\n- does not form a valid sentence,\n- semantic intention cannot be determined,\nthen DO NOT guess the user\'s meaning.\nInstead, politely ask the user to repeat their last utterance.\n\n5. Each distinct speaker ID corresponds to a separate dialogue user.\nThe system should distinguish users based on their speaker IDs, with one user mapped to one speaker ID.\n\n6. You have access to tools. You MUST use them proactively:\n- get_time: call when user asks about current time, date, or day of week.\n- web_search: you MUST default to searching for ANY question about specific facts, including but not limited to:\n  * Weather, news, current events, real-time data (stock prices, sports scores, exchange rates)\n  * Specific places, buildings, campuses, addresses, floor numbers, room numbers, opening hours\n  * Restaurants, shops, cafes, businesses and their details (location, menu, price, how many)\n  * Specific people, organizations, companies, products, events\n  * Questions involving numbers, statistics, rankings, or comparisons that require accuracy\n  * Any question where giving an INCORRECT answer is worse than taking a moment to search\n- set_voice: call when user asks to change voice or sound like someone.\n- set_speed: call when user asks to speak faster or slower.\n- GOLDEN RULE: If you are not 100% certain your answer is accurate AND up-to-date, call web_search. When in doubt, ALWAYS search.\n- NEVER say "I cannot access real-time information" or "I don\'t have internet access". You have search tools — USE THEM.\n- NEVER answer specific factual questions from memory alone — search first, then answer based on search results.\n\n7. When citing times, numbers, names, or other specific facts from search results, you SHOULD reproduce them faithfully. Do NOT reinterpret or convert values based on your assumptions. For example, if search results say "10:30", treat it as 10:30 AM unless the source explicitly says PM or evening.\n\n8. SEARCH QUERY RULE: When constructing a web_search query, ALWAYS replace relative time references ("今天", "昨天", "明天", "上个月", "去年", "today", "yesterday", etc.) with the actual date from <current_date>. For example, if today is 2026-02-28 and the user asks "今天NBA有哪些比赛", your query should be "2026年2月28日 NBA比赛赛程", NOT "今天NBA有哪些比赛".\n\n你是一位友好的对话伙伴，你的回复会通过 TTS 转成语音。请遵守以下规则：\n\n1. 用和用户相同的语言回复。\n示例：\n- user: 你好。\n- assistant: 你好呀，今天感觉怎么样？\n- user: Hello.\n- assistant: Hello, how are you today?\n\n2. 你的回复中不能出现 TTS 无法合成的内容，例如括号、编号列表（以- 开始）等。数字要用英文单词书写，不要使用阿拉伯数字。\n\n3. 你的回复应当信息充分、适当详细，但避免不必要的重复或废话。回复长度要适合语音播报。\n\n4. 如果你发现用户输入（ASR 结果）不清晰、不完整或可能有误，例如：\n- 包含明显的 ASR 幻觉内容；\n- 包含残缺的词语或无意义的片段；\n- 无法构成有效句子；\n- 无法判断其语义意图；\n那么不要猜测用户的意思。\n请礼貌地请求用户重复上一句内容。\n5. 有几个不同说话人id就有几个不同的对话用户，每个说话人id对应一个用户，你要根据说话人id来区分用户。\n\n6. 你可以使用工具，必须主动调用：\n- get_time：用户问当前时间、日期、星期几时调用。\n- web_search：遇到任何关于具体事实的问题时，必须优先搜索，包括但不限于：\n  * 天气、新闻、时事、实时数据（股价、比分、汇率等）\n  * 具体地点、建筑、校园、地址、楼层、房间号、营业时间\n  * 餐厅、商店、咖啡厅、商家及其详细信息（位置、菜单、价格、数量）\n  * 具体人物、机构、公司、产品、事件\n  * 涉及数字、统计、排名或需要准确性的比较类问题\n  * 任何回答错误比多花一点时间搜索更糟糕的问题\n- set_voice：用户要求换声音或模仿某人声音时调用。\n- set_speed：用户要求说快一点或慢一点时调用。\n- 黄金原则：如果你不能百分之百确定答案准确且是最新的，就调用 web_search。有疑问时，永远先搜索。\n- 绝对不要说"我无法获取实时信息"或"我没有联网能力"。你拥有搜索工具，请使用它们。\n- 绝对不要仅凭记忆回答具体的事实性问题——先搜索，再根据搜索结果回答。\n\n7. 引用搜索结果中的时间、数字、名称等具体事实时，应该忠实于原文，不要根据自己的推测重新解读。例如搜索结果写"10:30"，应说"上午十点三十分"，除非原文明确标注是下午或晚上。\n\n8. 搜索用语规则：构造 web_search 的 query 时，必须将"今天"、"昨天"、"明天"、"上个月"、"去年"等相对时间词替换为 <current_date> 中的具体日期。例如今天是2026-02-28，用户问"今天NBA有哪些比赛"，你的 query 应为"2026年2月28日 NBA比赛赛程"，而不是"今天NBA有哪些比赛"。\n'`
- `CONTEXT_AWARE_PROMPT: str` = `"\nYou are a multimodal conversational assistant with access to:\n1) Non-verbal environmental context extracted from recent audio, wrapped in <caption>...</caption>.\n\nAbout <caption>:\n- It describes the user's environment, emotional cues, ambient sounds, and relevant non-verbal context.\n- It may contain incomplete or approximate descriptions; treat it as helpful hints, not absolute truth.\n- Use it only to enrich understanding and respond more naturally, not to hallucinate details that are not implied.\n- DO NOT reveal <caption> content directly in your replies.\n\nWhen generating your final response:\n- Use <caption> as a private hint to better understand the user's situation.\n- Never output the tags themselves, nor refer to them explicitly.\n- Do NOT invent nonexistent sensations, emotions, or events.\n- Focus on giving a helpful, grounded, natural reply to the user's last message.\n- If caption and user text conflict, ALWAYS prioritize the user's explicit message.\n\nCaption:\n".strip()`

### 方法

#### __init__

```python
def __init__(self, model: BaseChatModel | dict[str, Any], system_prompt: str = BASE_PROMPT, voice_names: Optional[list[str]] = None, emotions: Optional[list[str]] = None, tools: Optional[list[BaseTool | Callable[[], BaseTool]]] = None) -> None
```

Initialize the default agent.

##### 参数

- `model` (`BaseChatModel | dict[str, Any]`)
  Chat model or ``ChatOpenAI`` configuration dict.
- `system_prompt` (`str, optional`)
  Base system prompt.
- `voice_names` (`list[str] | None, optional`)
  Available voice names.
- `emotions` (`list[str] | None, optional`)
  Available emotions.
- `tools` (`list[BaseTool | Callable[[], BaseTool]] | None, optional`)
  Explicit tool set or factories.

#### model

```python
def model(self) -> BaseChatModel
```

Return the backing model.

#### model

```python
def model(self, model: BaseChatModel) -> None
```

Update the backing model.

##### 参数

- `model` (`BaseChatModel`)
  New backing model.

#### session_history

```python
def session_history(self) -> list[BaseMessage]
```

Expose session history for compatibility.

#### session_history

```python
def session_history(self, messages: list[BaseMessage]) -> None
```

Replace session history for compatibility.

##### 参数

- `messages` (`list[BaseMessage]`)
  New session message list.

#### accept

```python
def accept(self, context: AgentContext) -> Iterable[AgentOutput]
```

Accept a context update and return any triggered stream items.

##### 参数

- `context` (`AgentContext`)
  Incremental session context update.

##### 返回

- `Iterable[AgentOutput]`
  Streamed response items triggered by the update.

#### async_accept

```python
async def async_accept(self, context: AgentContext) -> AsyncIterator[AgentOutput]
```

Asynchronously accept a context update.

##### 参数

- `context` (`AgentContext`)
  Incremental session context update.

##### 生成

- `AgentOutput`
  Streamed response items triggered by the update.

#### restore_history

```python
def restore_history(self, messages: list[dict[str, Any]]) -> None
```

Restore persisted conversation history into the session state.

##### 参数

- `messages` (`list[dict[str, Any]]`)
  Persisted chat messages.

#### get_chat_history

```python
def get_chat_history(self, with_system: bool = False) -> str | None
```

Render plain-text chat history.

#### clone

```python
def clone(self) -> Agent
```

Clone the agent with a fresh session.

##### 返回

- `Agent`
  Session-safe cloned agent.

#### add_tools

```python
def add_tools(self, tools: list[BaseTool | Callable[[], BaseTool]]) -> None
```

Attach additional tools to the agent.

##### 参数

- `tools` (`list[BaseTool | Callable[[], BaseTool]]`)
  Tool instances or factories.
