# Documentation Guidelines

## Bilingual Documentation

- Every documentation change or addition must cover both the English `.md` version and the Chinese `.zh.md` version.
- Every English document must have a corresponding Chinese document named `*.zh.md`.
- When creating or modifying an English document, create or update its Chinese counterpart as necessary.
- When creating or modifying a Chinese document, create or update its English counterpart as necessary.
- Keep corresponding English and Chinese documents structurally equivalent and factually consistent.

## Writing Style

- Avoid contrastive constructions equivalent to `not ... but ...` or the Chinese `不是……而是……` pattern.
- Prefer affirmative sentences throughout the documentation whenever possible.
- Minimize prose whenever possible. Prefer concise code blocks, commands, configuration snippets, and examples to explain technical content.

## Navigation and Links

- When adding, moving, renaming, or deleting a document, update `mkdocs.yml` as necessary.
- Keep English navigation labels and their Chinese entries in `nav_translations` synchronized.
- After moving or renaming documentation, update all affected references and verify relative Markdown links.

## Scope

- These instructions apply to all documentation under this directory, including tutorials, application tutorials, technical references, and API documentation.
