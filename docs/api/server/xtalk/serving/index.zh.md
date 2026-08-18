<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.serving

## ServiceManager

_定义于 [`xtalk.serving.service_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service_manager.py)。_

```python
class ServiceManager
```

Manage active Service instances for WebSocket sessions.

### 方法

#### __init__

_定义于 [`xtalk.serving.service_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service_manager.py)。_

```python
def __init__(self, models: Models | None = None, service_config: dict[str, Any] | None = None, service_prototype: Service | None = None, persistence_store: PersistenceStore | None = None)
```

#### create_service

_定义于 [`xtalk.serving.service_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service_manager.py)。_

```python
async def create_service(self, websocket: WebSocket, *, session_id: str | None = None, user_id: str | None = None) -> Service
```

Create a new Service instance bound to the given WebSocket.

#### remove_service

_定义于 [`xtalk.serving.service_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service_manager.py)。_

```python
async def remove_service(self, session_id: str) -> bool
```

Remove and shut down the Service with the given session id.

#### get_service

_定义于 [`xtalk.serving.service_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service_manager.py)。_

```python
def get_service(self, session_id: str) -> Optional[Service]
```

#### get_service_count

_定义于 [`xtalk.serving.service_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service_manager.py)。_

```python
def get_service_count(self) -> int
```

#### connect

_定义于 [`xtalk.serving.service_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/service_manager.py)。_

```python
async def connect(self, websocket: WebSocket, already_accepted: bool = False, user_id: str | None = None) -> None
```

Start a new service and connect to it for the given WebSocket.

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

- `MANAGER_CLASSES: list[Type[Manager]]` = `[ASRManager, MultiSpeakerTurnContextManager, LLMAgentContextManager, LLMAgentConsumptionManager, TTSManager, TTSResponseCoordinator, TTSPlaybackManager, CaptionerManager, RetrievalManager, TurnTakingManager, LatencyManager, VADManager, EnhancerManager, SpeakerManager, EmbeddingsManager, RecordingManager, TurnDetectorManager]`

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
async def publish(self, event: Event, mode: Union[EventDispatchMode, str] = EventDispatchMode.RETURN_AFTER_DISPATCH) -> bool
```

Publish an event to all matching handlers.

##### 参数

- `event` (`Event`)
  Event instance to dispatch.
- `mode` (`EventDispatchMode | str, optional`)
  Return and propagation behavior. Long canonical strings and the
  short aliases ``dispatch``, ``wait``, and ``wait_stoppable`` are
  accepted.

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

## EventDispatchMode

_定义于 [`xtalk.serving.event_bus`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/event_bus.py)。_

```python
class EventDispatchMode(str, Enum)
```

Control when :meth:`EventBus.publish` returns to its caller.

``RETURN_AFTER_DISPATCH`` preserves the default background-dispatch
behavior. The two waiting modes execute handlers in descending priority
order; only ``WAIT_UNTIL_COMPLETE_OR_STOPPED`` observes an explicit
:class:`EventPropagation.STOP` result.

### 类字段

- `RETURN_AFTER_DISPATCH` = `'return_after_dispatch'`
- `WAIT_UNTIL_COMPLETE` = `'wait_until_complete'`
- `WAIT_UNTIL_COMPLETE_OR_STOPPED` = `'wait_until_complete_or_stopped'`

### 方法

#### parse

_定义于 [`xtalk.serving.event_bus`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/event_bus.py)。_

```python
def parse(cls, value: Union['EventDispatchMode', str]) -> 'EventDispatchMode'
```

Normalize an enum member or supported long/short string.

##### 参数

- `value` (`EventDispatchMode | str`)
  Dispatch mode enum, canonical value, or short alias. Supported
  aliases are ``dispatch``, ``wait``, and ``wait_stoppable``.

##### 返回

- `EventDispatchMode`
  Canonical dispatch mode.

##### 抛出

- `ValueError`
  Raised when *value* is not a supported dispatch mode.

## EventPropagation

_定义于 [`xtalk.serving.event_bus`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/event_bus.py)。_

```python
class EventPropagation(str, Enum)
```

Describe whether a waiting event dispatch should continue.

### 类字段

- `CONTINUE` = `'continue'`
- `STOP` = `'stop'`
