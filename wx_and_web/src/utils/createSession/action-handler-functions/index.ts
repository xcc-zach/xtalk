import type { ActionToFunctionMap } from "./types";
export { ACTION_TO_FUNCTION };

const ACTION_TO_FUNCTION: ActionToFunctionMap = {}

function registerMap(partial_map: ActionToFunctionMap) {
    for (const key in partial_map) {
        if (partial_map[key]) {
            ACTION_TO_FUNCTION[key] = partial_map[key];
        }
    }
}

// ------------ Import and register action handler functions here ------------
import clientMap from "./client";
import messagesMap from "./messages";
import outputMap from "./output";
import sessionMap from "./session";
import clientOperationMap from "./client-operations";
import metaMap from "./meta";
import latencyMap from "./latency";
import inputMap from "./input";
registerMap(clientMap);
registerMap(messagesMap);
registerMap(outputMap);
registerMap(sessionMap);
registerMap(clientOperationMap);
registerMap(metaMap);
registerMap(latencyMap);
registerMap(inputMap);