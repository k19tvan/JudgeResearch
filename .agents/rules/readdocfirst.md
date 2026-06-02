---
trigger: always_on
---

---
description: Repository-wide development protocol and documentation workflow.
applyTo: "**"
# applyTo: 'Describe when these instructions should be loaded by the agent based on task context' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---

<!-- Tip: Use /create-instructions in chat to generate content with agent assistance -->
## Required Startup Context

Before starting any task, read:

- docs/AI_WORKFLOW.md
- docs/PROJECT_INDEX.md

Use them as the primary navigation system for understanding the repository.

# Documentation-First Development Protocol

This repository contains a structured documentation knowledge base under `/docs`.

When performing any task, follow this workflow.

## Required Reading Order

Before modifying code:

1. Read `docs/PROJECT_INDEX.md`
2. Identify the relevant module documentation
3. Read the relevant workflow documentation
4. Read the corresponding file documentation
5. Only then inspect and modify source code

Required navigation path:

```text
PROJECT_INDEX.md
    ↓
Relevant Module
    ↓
Relevant Workflow
    ↓
Relevant File Documentation
    ↓
Source Code
```

Never modify code without understanding the corresponding documentation.

---

## Documentation Mapping

For source files:

```text
src/backend/auth.py
↓
docs/files/src_backend_auth.md
```

Always read the matching documentation file first.

---

## Change Analysis Requirement

Before making modifications, determine:

* Relevant modules
* Relevant workflows
* Relevant source files
* Dependencies
* Potential impact

Use documentation as the primary source of truth.

---

## Documentation Maintenance

Documentation is part of the codebase.

Any behavioral, architectural, API, workflow, dependency, or implementation change must update documentation.

Documentation updates are mandatory.

---

## Update Sections

Never overwrite historical documentation.

Append changes under an `# Updates` section.

Example:

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

Create the section if it does not exist.

---

## File Documentation Updates

When modifying:

```text
src/example/file.py
```

Update:

```text
docs/files/src_example_file.md
```

Add:

* What changed
* Why it changed
* Affected functions
* Related files

---

## Module Documentation Updates

Update the relevant module document when:

* Responsibilities change
* New components are added
* Dependencies change
* Public APIs change

---

## Workflow Documentation Updates

Update workflow documents when:

* Execution order changes
* New steps are added
* Existing steps are removed
* Data flow changes

---

## Architecture Documentation Updates

Update architecture documents when:

* Component relationships change
* Data flow changes
* Execution flow changes
* New subsystems are introduced

---

## Final Verification Checklist

Before completing a task verify:

* Relevant documentation was reviewed
* Relevant workflow was reviewed
* Relevant file documentation was reviewed
* Source code changes are complete
* Documentation changes are complete
* Updates were added to the correct documentation files

If documentation is not updated, the task is not complete.

---

## Golden Rule

Documentation is treated as production code.

Every code change must leave the documentation more accurate than before.
