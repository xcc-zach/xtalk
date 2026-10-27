# X-Talk

> ⚠️ X-Talk is in active prototyping. Interfaces and functions are subject to change. We will try to keep interfaces stable.

X-Talk is an open-source framework for full-duplex spoken dialogue: a cascaded pipeline of ASR, LLM agent, TTS and turn-taking models that users can interrupt mid-sentence. The backend is pure Python with every model type pluggable from one script, the Web client SDK speaks an OpenAI Realtime style protocol over WebSocket, and a desktop app packages the same service for local use.

[Demo Link](https://xtalk.sjtuxlance.com/)

[Github](https://github.com/xcc-zach/xtalk)

<div class="grid cards next-step-cards" markdown>

-   **[Tutorial →](tutorial/start_the_service.md)** · Learn X-Talk step by step, from a running service to custom models and tools

-   **[How-to →](how_to/deploy_fully_local.md)** · Complete a specific task such as local deployment, logging or testing

-   **[Reference →](reference/service_config.md)** · Look up configuration keys, protocol events and model interfaces

-   **[Explanation →](explanation/system_design.md)** · Understand the architecture and why the pipeline behaves the way it does

</div>
