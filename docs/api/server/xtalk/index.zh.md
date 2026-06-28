<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk

## Xtalk

_定义于 [`xtalk.api`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/api.py)。_

```python
class Xtalk
```

Create Xtalk model services and session entrypoints.

### 说明

``Xtalk`` is the main integration surface used by the sample applications.
It builds model containers from configuration, stores a prototype service, and
accepts WebSocket sessions on demand.

### 方法

#### __init__

_定义于 [`xtalk.api`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/api.py)。_

```python
def __init__(self, *, service_prototype: Service, max_sessions: int | None = None)
```

Initialize an ``Xtalk`` application wrapper.

##### 参数

- `service_prototype` (`Service`)
  Prototype service used to clone per-session service instances.
- `max_sessions` (`int | None, optional`)
  Maximum number of concurrent sessions. If omitted, no session limit
  is enforced.

#### from_config

_定义于 [`xtalk.api`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/api.py)。_

```python
def from_config(cls, path_or_dict: str | dict) -> 'Xtalk'
```

Build an ``Xtalk`` instance from configuration data.

##### 参数

- `path_or_dict` (`str | dict`)
  JSON file path or already loaded configuration dictionary.

##### 返回

- `Xtalk`
  Configured application wrapper backed by a ``DefaultService``.

##### 示例

```pycon
>>> xtalk = Xtalk.from_config("server_config.json")
```

#### create_models_from_config

_定义于 [`xtalk.api`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/api.py)。_

```python
def create_models_from_config(cls, *, config_path_or_dict: str | dict, additional_models: dict[type[Any], Any] | None = None) -> Models
```

Instantiate configured models from configuration.

##### 参数

- `config_path_or_dict` (`str | dict`)
  JSON file path or already loaded configuration dictionary.
- `additional_models` (`dict[type[Any], Any] | None, optional`)
  Extra interface-to-instance mappings merged into the configured
  models.

##### 返回

- `Models`
  Model container created from the supplied configuration.

##### 示例

```pycon
>>> models = Xtalk.create_models_from_config(
...     config_path_or_dict="server_config.json",
...     additional_models={},
... )
```

#### set_session_limit

_定义于 [`xtalk.api`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/api.py)。_

```python
def set_session_limit(self, limit: int)
```

Set or replace the concurrent session limit.

##### 参数

- `limit` (`int`)
  Maximum number of active sessions allowed at the same time.

#### embed_text

_定义于 [`xtalk.api`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/api.py)。_

```python
async def embed_text(self, session_id: str, text: str, user_id: str | None = None)
```

Queue text for session-scoped embedding storage.

##### 参数

- `session_id` (`str`)
  Session identifier returned to the frontend.
- `text` (`str`)
  Text content that should be embedded and persisted for retrieval.

##### 抛出

- `ValueError`
  Raised if the target session does not exist.

#### add_agent_tools

_定义于 [`xtalk.api`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/api.py)。_

```python
def add_agent_tools(self, tools_or_factories: list[BaseTool | Callable[[], BaseTool]])
```

Attach tools to the prototype agent before sessions are created.

##### 参数

- `tools_or_factories` (`list[BaseTool | Callable[[], BaseTool]]`)
  Tool instances or zero-argument factories that produce tool
  instances.

##### 抛出

- `RuntimeError`
  Raised if at least one service session has already been created.

#### mount_routes

_定义于 [`xtalk.api`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/api.py)。_

```python
def mount_routes(self, app: Any, *, login_path: str = '/api/auth/login', sessions_path: str = '/api/sessions', session_detail_path: str = '/api/sessions/{session_id}', upload_path: str = '/api/upload', ws_path: str = '/ws') -> None
```

Mount the built-in auth, session, upload, and websocket routes.

#### connect

_定义于 [`xtalk.api`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/api.py)。_

```python
async def connect(self, websocket: WebSocket, user_id: str | None = None)
```

Accept a WebSocket session and hand it to the service manager.

##### 参数

- `websocket` (`WebSocket`)
  FastAPI WebSocket connection from the client.
- `user_id` (`str | None, optional`)
  Authenticated user identifier. When omitted, the connection falls
  back to the legacy connection-scoped session behavior.

##### 说明

If a session limit is configured, the socket is first admitted through
the session limiter queue.

## Models

_定义于 [`xtalk.models.container`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/container.py)。_

```python
class Models
```

Store model instances keyed by their model interface type.

### 方法

#### __init__

_定义于 [`xtalk.models.container`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/container.py)。_

```python
def __init__(self, entries: dict[type[Any], Any] | None = None) -> None
```

Initialize a model container.

##### 参数

- `entries` (`dict[type[Any], Any] | None, optional`)
  Initial interface-to-instance mapping.

#### get

_定义于 [`xtalk.models.container`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/container.py)。_

```python
def get(self, interface: type[T]) -> T | None
```

Return the model registered for an interface, if present.

##### 参数

- `interface` (`type[T]`)
  Model interface type used as the lookup key.

##### 返回

- `T | None`
  Registered model instance, or ``None`` when absent.

#### require

_定义于 [`xtalk.models.container`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/container.py)。_

```python
def require(self, interface: type[T]) -> T
```

Return a required model or raise a clear configuration error.

##### 参数

- `interface` (`type[T]`)
  Model interface type used as the lookup key.

##### 返回

- `T`
  Registered model instance.

##### 抛出

- `RuntimeError`
  Raised when no model is configured for ``interface``.

#### set

_定义于 [`xtalk.models.container`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/container.py)。_

```python
def set(self, interface: type[T], model: T | None) -> None
```

Set or remove a model for an interface.

##### 参数

- `interface` (`type[T]`)
  Model interface type used as the lookup key.
- `model` (`T | None`)
  Model instance to store. Passing ``None`` removes the mapping.

#### clone

_定义于 [`xtalk.models.container`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/container.py)。_

```python
def clone(self) -> 'Models'
```

Clone every cloneable model and share the rest.

## Service

_定义于 [`xtalk.serving.service`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service.py)。_

```python
class Service
```

Orchestrate a session-scoped model container and manager stack.

### 参数

- `models` (`Models`)
  Model container prototype that will be cloned for the session.
- `service_config` (`dict[str, Any] | None, optional`)
  Session configuration shared with managers and gateways.
- `manager_classes` (`list[Type[Manager]] | None, optional`)
  Manager classes to instantiate for live sessions.
- `_websocket` (`WebSocket | None, optional`)
  Internal WebSocket handle for live sessions. ``None`` means the instance
  acts as a prototype only.
- `_event_overrides` (`dict[Type[EventListenerMixin], EventOverrides] | None, optional`)
  Internal event subscription overrides copied into cloned sessions.

### 方法

#### __init__

_定义于 [`xtalk.serving.service`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service.py)。_

```python
def __init__(self, *, models: Models, service_config: dict[str, Any] | None = None, manager_classes: list[Type[Manager]] | None = None, _websocket: WebSocket | None = None, _session_id: str | None = None, _event_overrides: dict[Type[EventListenerMixin], EventOverrides] | None = None)
```

#### unsubscribe_event

_定义于 [`xtalk.serving.service`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service.py)。_

```python
def unsubscribe_event(self, *, event_listener_cls: Type[EventListenerMixin], event_type: Type[Event], method_name: str | None = None) -> None
```

Disable an automatic event subscription for a listener class.

##### 参数

- `event_listener_cls` (`Type[EventListenerMixin]`)
  Listener class whose subscription should be disabled.
- `event_type` (`Type[Event]`)
  Event type to unsubscribe.
- `method_name` (`str | None, optional`)
  Specific method name to disable. If omitted, every handler for the
  event is disabled for the listener class.

#### subscribe_event

_定义于 [`xtalk.serving.service`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service.py)。_

```python
def subscribe_event(self, *, event_listener_cls: Type[EventListenerMixin], event_type: Type[Event], method_or_handler: str | Callable[[EventListenerMixin, Event], Any] | Callable[[Event], Any] | Callable[[EventListenerMixin, Event], Coroutine[Any, Any, Any]] | Callable[[Event], Coroutine[Any, Any, Any]], priority: int = 0, enabled_if: Callable[[EventListenerMixin], bool] | None = None) -> None
```

Register an additional event subscription override.

##### 参数

- `event_listener_cls` (`Type[EventListenerMixin]`)
  Listener class that should receive the event.
- `event_type` (`Type[Event]`)
  Event type to subscribe to.
- `method_or_handler` (`str | Callable`)
  Method name on the listener instance or an external sync/async
  handler accepting ``event`` or ``(listener, event)``.
- `priority` (`int, optional`)
  Subscription priority. Higher values run first.
- `enabled_if` (`Callable[[EventListenerMixin], bool] | None, optional`)
  Predicate used to decide whether the subscription should be
  installed for a concrete listener instance.

#### register_manager

_定义于 [`xtalk.serving.service`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service.py)。_

```python
def register_manager(self, manager_cls: Type[Manager])
```

Register a manager class on the service.

##### 参数

- `manager_cls` (`Type[Manager]`)
  Manager class to add to the service prototype or live session.

#### unregister_manager

_定义于 [`xtalk.serving.service`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service.py)。_

```python
def unregister_manager(self, manager_cls: Type[Manager])
```

Remove a manager class from the service.

##### 参数

- `manager_cls` (`Type[Manager]`)
  Manager class to remove.

#### handle_message_loop

_定义于 [`xtalk.serving.service`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service.py)。_

```python
async def handle_message_loop(self, already_accepted: bool = False) -> None
```

Run the full WebSocket message loop for a live session.

##### 参数

- `already_accepted` (`bool, optional`)
  Whether the WebSocket has already been accepted by an upstream
  limiter or caller.

##### 抛出

- `RuntimeError`
  Raised if the service instance is still a prototype without runtime
  gateways.

#### restore_conversation

_定义于 [`xtalk.serving.service`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service.py)。_

```python
def restore_conversation(self, *, messages: list[dict[str, Any]]) -> None
```

Restore persisted conversation history into the session agent.

#### send_session_attached

_定义于 [`xtalk.serving.service`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service.py)。_

```python
async def send_session_attached(self) -> None
```

Notify the client that the session is attached.

#### stop

_定义于 [`xtalk.serving.service`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service.py)。_

```python
async def stop(self) -> None
```

Stop the service and shut down all managers.

#### clone

_定义于 [`xtalk.serving.service`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service.py)。_

```python
def clone(self, new_websocket: WebSocket, *, session_id: str | None = None, service_config_overrides: dict[str, Any] | None = None) -> 'Service'
```

Clone the service prototype for a new WebSocket session.

##### 参数

- `new_websocket` (`WebSocket`)
  WebSocket assigned to the new live session.

##### 返回

- `Service`
  Cloned service instance of the same concrete type.

## DefaultService

_定义于 [`xtalk.serving.service`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service.py)。_

```python
class DefaultService(Service)
```

Convenience ``Service`` with the standard Xtalk manager stack.

### 说明

Sample applications usually instantiate ``DefaultService`` directly and then
register or override managers for custom behavior.

### 类字段

- `MANAGER_CLASSES: list[Type[Manager]]` = `[ASRManager, LLMAgentContextManager, LLMAgentConsumptionManager, DirectAudioManager, TTSManager, TTSPlaybackManager, CaptionerManager, RetrievalManager, TurnTakingManager, LatencyManager, VADManager, EnhancerManager, SpeakerManager, EmbeddingsManager, RecordingManager, TurnDetectorManager]`

### 方法

#### __init__

_定义于 [`xtalk.serving.service`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service.py)。_

```python
def __init__(self, *, models: Models, service_config: dict[str, Any] | None = None, manager_classes: list[Type[Manager]] | None = None, _websocket: WebSocket | None = None, _session_id: str | None = None, _event_overrides: dict[Type[EventListenerMixin], EventOverrides] | None = None)
```

## Event

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class Event
```

Base dataclass for all Xtalk events.

### 参数

- `session_id` (`str`)
  Session identifier associated with the event.

### 属性

- `timestamp` (`float`)
  Unix timestamp recorded when the event instance is created.
- `session_id` (`str`)
  Session identifier associated with the event.
- `TYPE` (`str`)
  Stable event type string used by the event bus.

### 类字段

- `timestamp: float` = `field(init=False)`
- `session_id: str`
- `TYPE: ClassVar[str]` = `'base'`

### 方法

#### event_type

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
def event_type(self) -> str
```

## create_event_class

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
def create_event_class(*, name: str, fields: dict[str, Any] | None = None, type_name: str | None = None) -> Type[Event]
```

Create an ``Event`` subclass dynamically.

### 参数

- `name` (`str`)
  Dataclass name for the generated event type.
- `fields` (`dict[str, Any] | None, optional`)
  Mapping of field names to default values. Value types are inferred from
  the defaults.
- `type_name` (`str | None, optional`)
  Event bus type string. Defaults to ``name.lower()`` when omitted.

### 返回

- `Type[Event]`
  Generated dataclass type inheriting from ``Event``.

### 示例

```pycon
>>> CustomEvent = create_event_class(
...     name="CustomEvent",
...     fields={"text": ""},
... )
```

## Manager

_定义于 [`xtalk.serving.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/interfaces.py)。_

```python
class Manager(EventListenerMixin, ShutdownMixin)
```

Base class for Xtalk managers.

### 说明

Subclasses typically accept ``event_bus``, ``session_id``, ``models``, and
``config`` arguments, then register handlers with ``@Manager.event_handler``.

### 方法

#### event_handler

_定义于 [`xtalk.serving.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/interfaces.py)。_

```python
def event_handler(event_type: Type[Event], *, priority: int = 0, enabled_if: Callable[['EventListenerMixin'], bool] | None = None)
```

Declare a manager event handler.

##### 参数

- `event_type` (`Type[Event]`)
  Event class handled by the decorated method.
- `priority` (`int, optional`)
  Execution priority for the handler. Higher values run first.
- `enabled_if` (`Callable[[EventListenerMixin], bool] | None, optional`)
  Predicate evaluated against the manager instance before the handler
  is registered.

##### 返回

- `Callable`
  Decorator that marks the method for automatic subscription.

## EventBus

_定义于 [`xtalk.serving.event_bus`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/event_bus.py)。_

```python
class EventBus
```

Publish and subscribe session events with async dispatch support.

### 参数

- `enable_history` (`bool, optional`)
  Whether to store published events in memory for later inspection.
- `max_history` (`int, optional`)
  Maximum number of events kept when history is enabled.

### 类字段

- `MAX_ERROR_EVENT_DEPTH` = `3`
- `ERROR_EVENT_COOLDOWN` = `1.0`
- `ERROR_EVENT_RATE_LIMIT` = `10`

### 方法

#### __init__

_定义于 [`xtalk.serving.event_bus`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/event_bus.py)。_

```python
def __init__(self, enable_history: bool = False, max_history: int = 1000)
```

Initialize the event bus.

##### 参数

- `enable_history` (`bool, optional`)
  Whether to record published events in the in-memory history buffer.
- `max_history` (`int, optional`)
  Maximum number of events retained in history.

#### subscribe

_定义于 [`xtalk.serving.event_bus`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/event_bus.py)。_

```python
def subscribe(self, event_class: Union[Type[Event], str], handler: Callable[[Event], Any], priority: int = 0) -> None
```

Subscribe a handler to an event type.

##### 参数

- `event_class` (`Type[Event] | str`)
  Event class or event type string such as ``"tts.started"``.
- `handler` (`Callable[[Event], Any]`)
  Sync or async callable invoked for matching events.
- `priority` (`int, optional`)
  Higher values run earlier.

#### unsubscribe

_定义于 [`xtalk.serving.event_bus`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/event_bus.py)。_

```python
def unsubscribe(self, event_class: Union[Type[Event], str], handler: Callable) -> bool
```

Unsubscribe a handler from an event type.

##### 参数

- `event_class` (`Type[Event] | str`)
  Event class or event type string.
- `handler` (`Callable`)
  Previously subscribed handler to remove.

##### 返回

- `bool`
  ``True`` if the handler was removed, otherwise ``False``.

#### publish

_定义于 [`xtalk.serving.event_bus`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/event_bus.py)。_

```python
async def publish(self, event: Event, wait_for_completion: bool = False) -> bool
```

Publish an event to all matching handlers.

##### 参数

- `event` (`Event`)
  Event instance to dispatch.
- `wait_for_completion` (`bool, optional`)
  Whether to await handler completion before returning.

##### 返回

- `bool`
  ``True`` on success, otherwise ``False``.

#### get_history

_定义于 [`xtalk.serving.event_bus`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/event_bus.py)。_

```python
def get_history(self, event_type: Optional[str] = None, session_id: Optional[str] = None) -> List[Event]
```

Retrieve event history with optional filters.

Args:
    event_type: filter by type
    session_id: filter by session id

Returns:
    List of events

#### get_stats

_定义于 [`xtalk.serving.event_bus`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/event_bus.py)。_

```python
def get_stats(self) -> Dict[str, Any]
```

Return current event bus statistics.

Returns:
    Dict of stats

#### clear_history

_定义于 [`xtalk.serving.event_bus`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/event_bus.py)。_

```python
def clear_history(self) -> None
```

Clear event history.

#### reset_error_tracking

_定义于 [`xtalk.serving.event_bus`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/event_bus.py)。_

```python
def reset_error_tracking(self) -> None
```

Reset error event tracking (useful for testing).

#### shutdown

_定义于 [`xtalk.serving.event_bus`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/event_bus.py)。_

```python
async def shutdown(self) -> None
```

Shut down the event bus and release resources.

## model

_定义于 [`xtalk.models.registry`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/registry.py)。_

```python
def model(cls: type[Any] | None = None, *, name: str | None = None, aliases: list[str] | tuple[str, ...] | None = None, replace: bool = False) -> Callable[[type[Any]], type[Any]] | type[Any]
```

Register a model implementation class for configuration loading.

### 参数

- `cls` (`type[Any] | None, optional`)
  Implementation class when the decorator is used as ``@model``.
- `name` (`str | None, optional`)
  Canonical config name. Defaults to the class name.
- `aliases` (`list[str] | tuple[str, ...] | None, optional`)
  Additional accepted config names.
- `replace` (`bool, optional`)
  Whether an existing model registration may be replaced.

## model_type

_定义于 [`xtalk.models.registry`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/registry.py)。_

```python
def model_type(cls: type[Any] | None = None, *, aliases: list[str] | tuple[str, ...] | None = None, replace: bool = False) -> Callable[[type[Any]], type[Any]] | type[Any]
```

Register a model interface as a config-loadable model type.

### 参数

- `cls` (`type[Any] | None, optional`)
  Interface class when the decorator is used as ``@model_type``.
- `aliases` (`list[str] | tuple[str, ...] | None, optional`)
  Additional config keys accepted for backwards compatibility.
- `replace` (`bool, optional`)
  Whether an existing slot registration may be replaced.
