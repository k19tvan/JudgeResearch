# AI Workflow

This file defines the required workflow for AI agents working in this repository.

## Reading Order

Before making any change:

1. Read `docs/PROJECT_INDEX.md`
2. Identify the relevant module document
3. Read the relevant workflow document
4. Read the relevant file documentation
5. Inspect source code
6. Make changes
7. Update documentation

---

## Documentation Hierarchy

```text
PROJECT_INDEX.md
    ↓
Module Documentation
    ↓
Workflow Documentation
    ↓
File Documentation
    ↓
Source Code
```

---

## Source File Mapping

Example:

```text
src/backend/auth.py
↓
docs/files/src_backend_auth.md
```

Always read the documentation file before modifying the source file.

---

## Documentation Updates

Whenever code changes:

* Update the corresponding file documentation.
* Update the corresponding module documentation if responsibilities changed.
* Update workflow documentation if execution behavior changed.
* Update architecture documentation if system design changed.

Never leave documentation outdated.

---

## Update Section Format

Append updates under:

```markdown
# Updates

## YYYY-MM-DD

### Change Summary

...

### Files Modified

...

### Impact

...
```

Do not overwrite historical updates.

---

## Completion Checklist

Before finishing any task:

* Relevant documentation reviewed
* Relevant workflow reviewed
* Source code updated
* Documentation updated
* Update section added

If documentation is missing, the task is incomplete.
