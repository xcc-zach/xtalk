> [!NOTE]
> See `examples/sample_app/configurable_server.py` and `frontend/src/js/index.js` for details.
    
X-Talk can understand documents uploaded through embedding search. To enable embedding, you need `langchain_openai.OpenAIEmbeddings` in the config:
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

Then you can fetch `text` and `session_id` from client side and notify X-Talk instance through `embed_text`:
```python
@app.post("/api/upload")
async def upload_file(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    # Check file type
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    is_text = content_type.startswith("text/") if content_type else False
    if content_type and not is_text:
        raise HTTPException(status_code=400, detail="Only text files are supported.")
    # Read file content and embed
    text = (await file.read()).decode("utf-8", errors="ignore")
    await xtalk_instance.embed_text(session_id=session_id, text=text)
    return {"status": "ok"}
```
    
Note that client side should save `session_id` and send it in the request. Search `'session_info'` and `uploadFile` in `frontend/src/js/index.js` for how `session_id` is saved and used.