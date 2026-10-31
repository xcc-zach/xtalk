This is the frontend API bundle for X-Talk. See https://github.com/xcc-zach/xtalk for details.

## Build the package

Use `npm run build` when you want a production bundle in `dist/`. This is the command to run before publishing, packaging, or verifying the final output.

Use `npm run watch` while developing the frontend package locally. It runs webpack in watch mode and rebuilds automatically whenever files under `src/` change.

## Run the tests

`npm test` builds the bundle and runs `tests/**/*.test.ts` with the Node test
runner (Node 22.18 or newer). The tests drive the public `createSession` API
against the repository's Python mock backend, so install the backend first from
the repository root with `pip install -e ".[test]"`. Set `XTALK_PYTHON` when the
interpreter is not `python3`.

## Generate API documentation

The frontend API docs are generated with TypeDoc and `typedoc-plugin-markdown`, then written to the repository docs site so MkDocs can include them.

From the repository root:

```bash
cd frontend
npm install
npm run docs
```

The generated Markdown files are written to `../docs/api/client/`.

Only the English API docs are generated. The MkDocs i18n configuration falls back to the English page for the Chinese site.
