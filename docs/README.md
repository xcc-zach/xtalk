# Docs Build

This repository keeps API docs as prebuilt Markdown files under `docs/api/`.
MkDocs reads those files directly and does not generate the API reference pages
during site build.

## Frontend API docs

From the repository root:

```bash
cd frontend
npm install
npm run docs
```

The generated Markdown files are written to `docs/api/client/`.

## Backend API docs

From the repository root:

```bash
python docs/generate_server_docs.py
```

The backend generator writes Markdown files to `docs/api/server/` for the
Xtalk modules imported by [`examples/sample_app/*.py`](https://github.com/xcc-zach/xtalk/tree/main/examples/sample_app).

Only the English backend API docs are generated. The MkDocs i18n
configuration falls back to the English pages for the Chinese site.
