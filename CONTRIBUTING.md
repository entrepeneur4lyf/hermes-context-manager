# Contributing to Hermes Context Manager

Thank you for your interest in contributing. HMC is maintained solo and
the time I can dedicate to reviewing contributions is limited, so please
read this document before opening an issue or PR. Contributions that
follow these guidelines get reviewed faster.

## Before you open an issue

- **Search existing issues and discussions first.** Duplicates are closed
  without comment.
- **Questions belong in [Discussions](https://github.com/entrepeneur4lyf/hermes-context-manager/discussions), not Issues.**
  Issues are for reproducible bugs and concrete feature proposals.
- **Bug reports require a reproduction.** Use the bug report template and
  fill in every required field. "It doesn't work" is not actionable.
- **Feature requests require a problem statement.** Describe the real
  workload or use case. Features framed as "it would be cool if" without
  a concrete motivation get deferred indefinitely.

## Before you open a PR

1. **Open an issue first for non-trivial changes.** A PR that fixes a
   typo or a small bug can go straight in. A PR that adds a new
   compression strategy, changes the hook architecture, or introduces
   a new dependency needs discussion in an issue first. Unsolicited
   large refactor PRs will be closed.

2. **Run the full test suite and make sure it's green:**
   ```bash
   python -m unittest discover -s tests
   ```
   PRs that don't pass CI will not be reviewed until they do.

3. **Add tests for any code change.** HMC's test suite covers the
   strategy pipeline, hook handlers, persistence, analytics, and
   dashboard endpoints. Every bug fix should have a regression test
   that would have caught the bug. Every new feature should have
   tests covering the new behavior and at least one edge case.

4. **Follow the existing code style.** No new linters or formatters
   are configured; match the surrounding code. Key conventions:
   - Dataclasses with `slots=True` for state objects
   - Broad `try/except` around hook entry points so plugin bugs
     never crash Hermes
   - Docstrings explain *why*, not *what* — the code tells you what
   - `LOGGER.exception(...)` for failure paths, never silent `except`

5. **Do not add third-party dependencies without discussion.** HMC is
   stdlib + PyYAML only by design. New dependencies need justification
   in an issue first.

6. **Write a useful commit message.** See existing commits for the
   style — subject line describes the change, body explains the why,
   multi-paragraph is fine.

## Development setup

```bash
git clone https://github.com/entrepeneur4lyf/hermes-context-manager.git
cd hermes-context-manager
python -m unittest discover -s tests -v
```

No virtualenv, no build step — HMC is stdlib + PyYAML. If you're
testing against a live Hermes instance:

```bash
hermes plugins install file:///absolute/path/to/hermes-context-manager
hermes gateway restart
```

## Branch and review process

- `main` is protected. All changes go through pull requests.
- CI must be green on all supported Python versions (3.11, 3.12, 3.13).
- At least one reviewer approval is required before merge.
- Stale approvals are dismissed when new commits are pushed.
- All conversations must be resolved before merge.
- Force pushes to `main` are blocked.

## What gets accepted

- **Bug fixes** with a regression test: almost always accepted.
- **New compression strategies** with tests and a clear use case:
  reviewed carefully.
- **Dashboard improvements** (new panels, new endpoints): welcome,
  but must not break existing API shapes.
- **Documentation fixes**: always welcome, no discussion needed.
- **Test coverage improvements**: always welcome.

## What doesn't get accepted

- **Refactors for refactoring's sake.** If your PR description is "this
  code is cleaner now," it will be closed. Refactors need to enable a
  concrete feature or fix a concrete bug.
- **New dependencies without discussion.** See above.
- **Config renames or breaking API changes** without a deprecation path.
- **Behavioral changes that aren't tested.**

## Licensing

By contributing, you agree that your contributions will be licensed
under the same terms as the rest of the project. See `LICENSE`.
