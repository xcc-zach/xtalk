"""Compare current payloads to fixtures executed against the pre-migration gateway."""

import asyncio
import json
from pathlib import Path

import pytest
from xtalk import Models, Service
from xtalk.serving import events
from xtalk.serving.event_bus import EventDispatchMode

BASELINE = json.loads(
    (Path(__file__).parent / "fixtures/migration/gateway.json").read_text()
)


@pytest.mark.parametrize(
    "case", BASELINE["cases"], ids=lambda case: case["event_class"]
)
def test_original_snapshots_preserve_fields_and_replace_the_wire_envelope(case):
    """Retain legacy user-visible payloads across the approved envelope change."""

    async def run():
        class Transport:
            """Collect protocol payloads from a model-free session."""

            def __init__(self):
                self.messages = []

            async def send_json(self, value):
                """Capture one outgoing event."""
                self.messages.append(value)

        prototype = Service(models=Models())
        transport = Transport()
        service = prototype.clone(transport)
        event = vars(events)[case["event_class"]](
            session_id="session", **case["fields"]
        )
        await service.event_bus.publish(
            event, mode=EventDispatchMode.WAIT_UNTIL_COMPLETE
        )
        assert len(transport.messages) == 1
        actual = dict(transport.messages[0])
        assert actual.pop("event_id")
        assert actual == {"type": case["realtime_type"], **case["legacy"]["data"]}
        await service.stop()
        await prototype.stop()

    asyncio.run(run())
