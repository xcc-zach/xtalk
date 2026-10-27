# Voice Wake

This guide turns on hands-free conversation start in the X-Talk desktop app; it assumes the app already runs with a model configuration from [Model Configuration](../../tutorial/desktop_app/model_configuration.md).

When voice wake is enabled, the X-Talk desktop app listens for the wake phrase while no conversation is active. After detecting the phrase, the app starts a new voice conversation. Wake-phrase listening pauses during the conversation and resumes after it ends.

## Configure voice wake

1. Open **Settings and diagnostics** in the upper-right corner.
2. Expand **Voice wake**.
3. Enable **Voice wake**.
4. Enter a Chinese or English phrase under **Wake phrase**.
5. Set **Threshold** to a value from `0` to `1`.

A higher threshold makes detection stricter. A lower threshold makes wake detection more sensitive but may cause more false activations. The app automatically updates its listener after the wake phrase or threshold changes.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Web Search →](web_search.md)** · How-to

-   **[Add Tools →](../../tutorial/desktop_app/add_tools.md)** · Tutorial

</div>
