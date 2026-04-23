> [!NOTE]
> See `examples/sample_app/configurable_server.py`, `frontend/src`, `examples/sample_app/templates` and `examples/sample_app/static` for details.
   
X-Talk has most models and execution on server side, and the client is responsible for interacting with microphone, transmitting audio and Websocket messages, and handle lightweight operations like Voice-Actitvty-Detection.
    
For client side, you can start with snippet in `examples\sample_app\static\js\index.js` and track where `convo` is used to see how to use the client API:
```javascript
async function loadXtalk() {
    try {
        return await import("../../xtalk/index.js"); // Try local import first, this is dev only
    } catch (e) {
        return await import("https://unpkg.com/xtalk-client@latest/dist/index.js"); // Use unpkg CDN for production
    }
}

const { createConversation } = await loadXtalk();


function getWebSocketURL() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const wsPath = new URL("./ws", window.location.href);
    wsPath.protocol = proto;
    wsPath.host = window.location.host;
    return wsPath
}

const convo = createConversation(getWebSocketURL());
```
We recently published the client API as a separate package [xtalk-client](https://www.npmjs.com/package/xtalk-client). Therefore, you can directly import it from `https://unpkg.com/xtalk-client@latest/dist/index.js` without hosting the client code by yourself, as shown above. We plan to continuously improve the client-side API in the future.

For the server side, the core logic is to connect a X-Talk instance to Websocket of FastAPI instance:
```python
from fastapi import FastAPI, WebSocket
from xtalk import Xtalk
app = FastAPI(title="Xtalk Server")
xtalk_instance = Xtalk.from_config("path/to/config.json")
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await xtalk_instance.connect(websocket)
```
    
Then you can check `examples/sample_app/configurable_server.py` for how to mount client-side scripts and pages.