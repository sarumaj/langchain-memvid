# Changelog

All notable changes to this project are documented in this file. The release bot
writes each version's section from the commits (Conventional Commits) and the hand-written
Unreleased section.

## Unreleased

### Fixed

- 0.1.11 was tagged but not published: its logging tests failed under pytest 9.1, which
  stopped the release before PyPI. This version publishes its changes, with the tests fixed.

## 0.1.11 - 2026-11-09

### Dependencies

- update dependency pydantic to v2.14.3
- update dependency pytest-asyncio to v1.4.1.post1
- update dependency pytest-benchmark to v5.3.1
- update all patch dependencies
- update dependency langchain-core to v1.6.13
- update dependency sphinx-autodoc-typehints to <3.13.11

