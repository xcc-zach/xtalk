from abc import ABC, abstractmethod, ABCMeta
from typing import Type, Callable, Any
from dataclasses import dataclass, field
from inspect import signature, isawaitable
from .events import BaseEvent
from .event_bus import EventBus


@dataclass
class EventOverrides:
    disable: list[tuple[str, Type[BaseEvent]]] = field(default_factory=list)
    extra: list[dict[str, Any]] = field(default_factory=list)

    def should_skip(self, method_name: str, event_type: Type[BaseEvent]) -> bool:
        if any(m == "*" and et is event_type for m, et in self.disable):
            return True
        return any(m == method_name and et is event_type for m, et in self.disable)

    def register_extra(self, obj: Any, event_bus: EventBus) -> None:
        for item in self.extra:
            ev_type: Type[BaseEvent] = item["event_type"]
            priority: int = item.get("priority", 0)
            enabled_if = item.get("enabled_if", None)

            if enabled_if is not None and not enabled_if(obj):
                continue

            # Case 1: method defined on the class (method_name)
            if "method_name" in item:
                handler = getattr(obj, item["method_name"])
                event_bus.subscribe(ev_type, handler, priority=priority)
                continue

            # Case 2: external handler callable
            raw_handler = item["handler"]
            sig = signature(raw_handler)
            params = list(sig.parameters.values())

            # Supports sync/async + one/two-argument signatures:
            #   def h(event) / async def h(event)
            #   def h(manager, event) / async def h(manager, event)
            if len(params) == 1:
                # handler(event)
                async def wrapper(event, _h=raw_handler):
                    res = _h(event)
                    if isawaitable(res):
                        return await res
                    return res

            else:
                # handler(manager, event)
                async def wrapper(event, _h=raw_handler, _mgr=obj):
                    res = _h(_mgr, event)
                    if isawaitable(res):
                        return await res
                    return res

            event_bus.subscribe(ev_type, wrapper, priority=priority)


class EventListenerMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        handlers_meta: list[tuple[str, list[dict[str, Any]]]] = []
        for attr_name, attr in namespace.items():
            meta_list = getattr(attr, "__event_handlers__", None)
            if meta_list:
                handlers_meta.append((attr_name, meta_list))

        cls.__event_handlers_meta__ = handlers_meta
        return cls

    def __call__(cls, *args, **kwargs):
        overrides: EventOverrides | None = kwargs.pop("_event_overrides", None)
        obj = super().__call__(*args, **kwargs)
        event_bus: EventBus = getattr(obj, "event_bus", None)
        no_event_bus = False
        if event_bus is None:
            no_event_bus = True

        for base in reversed(obj.__class__.mro()):
            handlers_meta = getattr(base, "__event_handlers_meta__", [])
            for method_name, meta_list in handlers_meta:
                if no_event_bus:
                    raise RuntimeError(
                        f"Cannot register event handler '{method_name}' without an event bus."
                    )
                method = getattr(obj, method_name)
                for meta in meta_list:
                    ev_type: Type[BaseEvent] = meta["event_type"]
                    priority: int = meta["priority"]
                    enabled_if = meta["enabled_if"]

                    if enabled_if is not None and not enabled_if(obj):
                        continue

                    if overrides is not None and overrides.should_skip(
                        method_name, ev_type
                    ):
                        continue

                    event_bus.subscribe(ev_type, method, priority=priority)
        if overrides is not None:
            overrides.register_extra(obj, event_bus)
        return obj


class EventListenerMixin(metaclass=EventListenerMeta):
    @staticmethod
    def event_handler(
        event_type: Type[BaseEvent],
        *,
        priority: int = 0,
        enabled_if: Callable[["EventListenerMixin"], bool] | None = None,
    ):
        def decorator(func: Callable[[BaseEvent], Any]):
            meta_list = getattr(func, "__event_handlers__", [])
            meta_list.append(
                {
                    "event_type": event_type,
                    "priority": priority,
                    "enabled_if": enabled_if,
                }
            )
            setattr(func, "__event_handlers__", meta_list)
            return func

        return decorator


class ShutdownMixin(ABC):
    @abstractmethod
    def shutdown(self):
        pass


class Manager(EventListenerMixin, ShutdownMixin):
    """Base class for xtalk managers that subscribe to events and clean up resources."""
