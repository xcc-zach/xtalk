# Voice Wake

When voice wake is enabled, the X-Talk desktop app listens for the wake phrase while no conversation is active. After detecting the phrase, the app starts a new voice conversation. Wake-phrase listening pauses during the conversation and resumes after it ends.

## Configure voice wake

1. Open **Settings and diagnostics** in the upper-right corner.
2. Expand **Voice wake**.
3. Enable **Voice wake**.
4. Enter a Chinese or English phrase under **Wake phrase**.
5. Set **Threshold** to a value from `0` to `1`.

A higher threshold makes detection stricter. A lower threshold makes wake detection more sensitive but may cause more false activations. The app automatically updates its listener after the wake phrase or threshold changes.
