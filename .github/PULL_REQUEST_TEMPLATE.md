<!--
Thanks for contributing to HMC. Before submitting, please confirm the checklist below.
PRs that don't include a reason, tests, or meaningful description will not be reviewed.
-->

## What does this PR do?

<!-- One-paragraph summary. If this is a bug fix, link the issue. -->

## Why?

<!-- What problem does it solve? What's the real-world workload where this matters?
For bug fixes, describe the faulty behavior and how you verified the fix.
For features, explain the use case. -->

## Testing

<!-- How did you verify this works? New tests added? Existing tests still pass?
HMC requires tests for any code change that touches the strategy pipeline, hook handlers,
or the dashboard endpoints. "Manual testing" alone is not sufficient for merge. -->

## Checklist

- [ ] Tests added for new behavior (or: no new behavior to test)
- [ ] Full test suite passes locally: `python -m unittest discover -s tests`
- [ ] README / CLAUDE.md updated if user-facing behavior or architecture changed
- [ ] Commit messages describe the change, not just "fix bug" or "update"
- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md)
