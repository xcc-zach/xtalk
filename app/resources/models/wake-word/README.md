# Wake-word model staging

`prepare_managed_runtime.py` copies the selected sherpa-onnx keyword-spotting
encoder, decoder, joiner, and token table into this directory. The checked-in
`keywords.txt` configures the fixed desktop wake phrase `你好小克`.

Use the `sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20` chunk-16 model
layout. Model weights remain prepared build artifacts and are not committed
here.
