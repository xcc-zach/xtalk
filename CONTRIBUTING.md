Thanks for considering contributing! 🎉

X-Talk is currently in an early **prototype stage** and is under active and rapid development.  
APIs, internal abstractions, and module boundaries may change as the project evolves.

Despite this, **contributions are highly welcome**.

We strongly encourage the community to:
- Experiment with the framework and report issues
- Propose design improvements or architectural suggestions
- Contribute new models, managers, or tools

## Branching Strategy

- `main`  
  Stable branch. Always kept in a runnable and relatively stable state.

- `dev`  
  Active development branch. New features, refactors, and most bug fixes
  are merged here first.

## Contributing with Pull Requests

👉 **Please open all pull requests against the `dev` branch.**

The `main` branch is updated periodically by maintainers
(e.g. during releases or milestone merges).

Pull requests opened against `main` may be asked to retarget `dev`.

Pull requests are reviewed **actively and constructively**, and we aim to merge high-quality contributions in a timely manner.  
Early contributors will have a meaningful impact on shaping the core design and direction of X-Talk.

If you are unsure whether an idea fits the current roadmap, feel free to open an issue or start a discussion — we value early feedback and collaborative iteration.

### Development Workflow

1. **Fork** the repository to your GitHub account.

2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/xtalk.git
   cd xtalk
   ```

3. **Sync and switch to the `dev` branch**:

   ```bash
   git checkout dev
   git pull origin dev
   ```

4. **Create a feature branch from `dev`**:

   ```bash
   git checkout -b feat/<short-description>
   ```

   Examples:

   * `feat/cosyvoice-streaming`
   * `fix/tts-queue-race`
   * `docs/quickstart-update`

5. **Make your changes** and commit them with clear messages:

   ```bash
   git commit -m "feat: add streaming TTS backend"
   ```

6. **Push** your branch to your fork:

   ```bash
   git push origin feat/<short-description>
   ```

7. **Open a Pull Request targeting the `dev` branch**.

   * Please ensure the base branch is set to `dev`, not `main`.
   * PRs targeting `main` may be asked to retarget `dev`.

## Running the Tests

Both suites exercise the real conversation pipeline with mock models, so they
run without GPUs, model services or sound cards.

Backend (from the repository root):

```bash
pip install -e ".[test]"
python -m pytest
```

Frontend (requires Node 22.18 or newer and the Python backend installed as
above, because the tests start `python -m tests.support.server` as a mock
backend; set `XTALK_PYTHON` to pick the interpreter):

```bash
cd frontend
npm ci
npm test
```

`npm test` builds `dist/` first and then runs `tests/**/*.test.ts` with the
Node test runner.

## Licensing

By contributing, you agree that your contributions will be licensed under the **Apache License 2.0**, consistent with the project license.
