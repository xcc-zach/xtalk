export { Conversation };
export type { ConversationMessage, ConversationState, ConversationUser };

type ConversationMessage = {
    /** Source role used for display and incremental update behavior. */
    role: "user" | "assistant" | "info";
    /** Plain-text message body. */
    content: string;
    /** Whether this message no longer accepts incremental updates. */
    final?: boolean;
    /** Stable server identifier for an assistant response. */
    responseId?: string;
}

type ConversationUser = {
    id: string;
}

function defaultConversation(): {
    connectionState: "connected" | "reconnecting" | "disconnected";
    streamState: "idle" | "listening" | "processing" | "speaking";
    sessionId: string | null;
    user: ConversationUser | null;
    latency: {
        network?: number,
        asr?: number,
        llmFirstToken?: number,
        llmSentence?: number,
        ttsFirstChunk?: number
    };
    messages: ConversationMessage[];
    thought: string;
    caption: string;
    retrieval: string;
    tool_call: {
        name: string,
        args: Record<string, any>
    }
} {
    return {
        connectionState: "disconnected",
        streamState: "idle",
        sessionId: null,
        user: null,
        latency: {},
        messages: [],
        thought: "",
        caption: "",
        retrieval: "",
        tool_call: {
            name: "",
            args: {}
        }
    };
}

type ConversationState = ReturnType<typeof defaultConversation>;

class Conversation {
    private _state: ConversationState = defaultConversation();
    private messagePrefixes: Array<string | undefined> = [];
    private assistantResponseIndexes = new Map<string, number>();
    private stateChangeCallbacks = new Set<(state: ConversationState) => void>();
    private fullAudioChunkCallback: (pcmChunkInt16: ArrayBuffer, sampleRate: number) => void = (_chunk, _sr) => { };

    private notifyStateChange(): void {
        for (const callback of this.stateChangeCallbacks) {
            callback(this._state);
        }
    }

    onStateChange(callback: (state: ConversationState) => void): void {
        callback(this._state);
        this.stateChangeCallbacks.add(callback);
    }

    onFullAudioChunk(
        callback: (pcmChunkInt16: ArrayBuffer, sampleRate: number) => void
    ): void {
        this.fullAudioChunkCallback = callback;
    }

    get state(): ConversationState {
        return new Proxy(this._state, {
            set: (target, key: keyof ConversationState, value) => {
                Reflect.set(target, key, value);
                this.notifyStateChange();
                return true;
            },
            get: (target, key: keyof ConversationState) => {
                return key in target ? target[key] : undefined;
            }
        });
    }

    setUser(user: ConversationUser | null): void {
        this._state.user = user;
        this.notifyStateChange();
    }

    switch(sessionId: string | null, messages: ConversationMessage[]): void {
        this._state.sessionId = sessionId;
        this._state.messages = messages.map((message) => ({
            ...message,
            final: true,
        }));
        this.messagePrefixes = this._state.messages.map(() => undefined);
        this.assistantResponseIndexes.clear();
        this._state.messages.forEach((message, index) => {
            if (message.role === "assistant" && message.responseId) {
                this.assistantResponseIndexes.set(message.responseId, index);
            }
        });
        this._state.streamState = "idle";
        this._state.thought = "";
        this._state.caption = "";
        this._state.retrieval = "";
        this._state.tool_call = {
            name: "",
            args: {}
        };
        this._state.latency = {};
        this.notifyStateChange();
    }

    appendMessage(message: ConversationMessage): void {
        if (message.role === "info") {
            this._state.messages.push(message);
            this.messagePrefixes.push(undefined);
            this.notifyStateChange();
            return;
        }

        const final = message.final ?? false;
        const lastIndex = this._state.messages.length - 1;
        const lastMessage = this._state.messages[lastIndex];

        if (lastMessage?.role === message.role && !lastMessage.final) {
            const prefix = this.messagePrefixes[lastIndex];
            const fullContent = `${prefix ?? ""}${lastMessage.content}`;
            if (message.role === "assistant") {
                if (message.content.startsWith(fullContent)) {
                    lastMessage.content = prefix
                        ? message.content.slice(prefix.length)
                        : message.content;
                    lastMessage.final = final;
                    this.notifyStateChange();
                    return;
                }
                if (fullContent.startsWith(message.content)) {
                    if (final) {
                        lastMessage.final = true;
                        this.messagePrefixes[lastIndex] = undefined;
                    }
                    this.notifyStateChange();
                    return;
                }

                // A non-prefix assistant update belongs to a new turn. Preserve
                // the playback-confirmed text from the previous turn instead of
                // replacing it merely because its finish signal was delayed.
                lastMessage.final = true;
                this.messagePrefixes[lastIndex] = undefined;
            } else {
                if (prefix && message.content.startsWith(prefix)) {
                    lastMessage.content = message.content.slice(prefix.length);
                } else {
                    lastMessage.content = message.content;
                    this.messagePrefixes[lastIndex] = undefined;
                }
                lastMessage.final = final;
                this.notifyStateChange();
                return;
            }
        }

        const previousSameRole = this.findPreviousSameRoleMessage(message.role);
        let content = message.content;
        let prefix: string | undefined;

        if (
            previousSameRole &&
            !previousSameRole.message.final &&
            message.content.startsWith(previousSameRole.fullContent)
        ) {
            prefix = previousSameRole.fullContent;
            content = message.content.slice(prefix.length);
            if (final) {
                previousSameRole.message.final = true;
            }
            if (!content) {
                this.notifyStateChange();
                return;
            }
        }

        this._state.messages.push({
            role: message.role,
            content,
            final,
        });
        this.messagePrefixes.push(prefix);
        this.notifyStateChange();
    }

    /**
     * Replace the cumulative text of one identified assistant response.
     *
     * A response ID remains stable across incremental and final updates, so
     * interleaved tool responses never rely on prefix or adjacency inference.
     *
     * @param responseId Stable server-assigned assistant response identifier.
     * @param content Full playback-confirmed response text.
     * @param final Whether this update closes the response.
     */
    updateAssistantResponse(
        responseId: string,
        content: string,
        final: boolean,
    ): void {
        if (!responseId) {
            return;
        }
        const existingIndex = this.assistantResponseIndexes.get(responseId);
        if (existingIndex !== undefined) {
            const message = this._state.messages[existingIndex];
            if (!message || message.role !== "assistant" || (message.final && !final)) {
                return;
            }
            message.content = content;
            message.final = final;
            this.notifyStateChange();
            return;
        }
        if (!content) {
            return;
        }
        const messageIndex = this._state.messages.length;
        this._state.messages.push({
            role: "assistant",
            responseId,
            content,
            final,
        });
        this.messagePrefixes.push(undefined);
        this.assistantResponseIndexes.set(responseId, messageIndex);
        this.notifyStateChange();
    }

    /**
     * Mark unfinished messages as final after a turn boundary or stream failure.
     *
     * This prevents stale streaming cursors when the server cannot emit the
     * normal role-specific finish action, for example after empty TTS output.
     *
     * @param role Optional role used to limit which pending messages are closed.
     */
    finalizePendingMessages(role?: ConversationMessage["role"]): void {
        let changed = false;
        for (let index = 0; index < this._state.messages.length; index++) {
            const message = this._state.messages[index];
            if (message && !message.final && (!role || message.role === role)) {
                message.final = true;
                this.messagePrefixes[index] = undefined;
                changed = true;
            }
        }
        if (changed) {
            this.notifyStateChange();
        }
    }

    private findPreviousSameRoleMessage(
        role: Exclude<ConversationMessage["role"], "info">,
    ): { message: ConversationMessage; fullContent: string } | undefined {
        for (let i = this._state.messages.length - 1; i >= 0; i--) {
            const message = this._state.messages[i];
            if (message?.role === role) {
                return {
                    message,
                    fullContent: `${this.messagePrefixes[i] ?? ""}${message.content}`,
                };
            }
        }
        return undefined;
    }

    updateLatency(latency: Conversation["state"]["latency"]): void {
        this._state.latency = { ...latency };
        this.notifyStateChange();
    }

    emitFullAudioChunk(pcmChunkInt16: ArrayBuffer, sampleRate: number): void {
        this.fullAudioChunkCallback(pcmChunkInt16, sampleRate);
    }
}
