> [!NOTE]
> 详情请参阅 `examples/sample_app/configurable_server.py` 和 `frontend/src/js/index.js`。

X-Talk 可以通过嵌入检索来理解用户上传的文档。要启用嵌入功能，您需要在配置中使用 `langchain_openai.OpenAIEmbeddings`：
```json
"embeddings": {
    "type": "OpenAIEmbeddings",
    "params": {
      "api_key": "<API_KEY>",
      "base_url": "<URL LIKE http://127.0.0.1:8002/v1>",
      "model": "<MODEL LIKE Qwen/Qwen3-Embedding-0.6B>"
    }
  },
```

然后，您可以从客户端获取 `text` 和 `session_id`，并通过 `embed_text` 通知 X-Talk 实例：
```python
@app.post("/api/upload")
async def upload_file(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    # 检查文件类型
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    is_text = content_type.startswith("text/") if content_type else False
    if content_type and not is_text:
        raise HTTPException(status_code=400, detail="Only text files are supported.")
    # 读取文件内容并执行嵌入
    text = (await file.read()).decode("utf-8", errors="ignore")
    await xtalk_instance.embed_text(session_id=session_id, text=text)
    return {"status": "ok"}
```

请注意，客户端需要保存 `session_id` 并在请求中发送它。您可以在 `frontend/src/js/index.js` 中搜索 `session_info` 和 `uploadFile`，查看 `session_id` 是如何保存和使用的。
