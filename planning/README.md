# Planning Directory

Working plans and task breakdowns for active development work.

---

## Purpose

This directory contains **ephemeral planning documents** that guide current implementation work. Unlike `docs/` which contains permanent technical documentation, plans here are:
- **Working documents** - Actively updated as work progresses
- **Time-bound** - Relevant for current phase/milestone only
- **Regularly cleaned** - Archived or deleted when complete/obsolete

---

## Current Plans

### IMPLEMENTATION_PLAN.md
7-phase plan for OSDI-based code generation restart (Jan 2026).

**Status**: Active - Phase 1 starting
**Archive when**: All phases complete (Phase 7 done)

---

## Guidelines

### What Goes Here
- Implementation plans with phase breakdowns
- Task lists for specific features or milestones
- Investigation plans for debugging efforts
- Roadmaps for major architectural changes

### What Doesn't Go Here
- Permanent technical documentation → `docs/`
- API references → `docs/`
- Architecture overviews → `docs/`
- Code examples → `docs/` or inline code comments

### Cleanup Policy

**Review regularly** (at least monthly or when completing major milestones):

1. **Completed plans** → Archive to `planning/archive/YYYY-MM-name/` with completion summary
2. **Obsolete plans** → Delete entirely if superseded or no longer relevant
3. **Stale plans** → Update or delete if not touched in 2+ months

**When to archive vs delete**:
- Archive if plan succeeded and provides historical context
- Delete if plan was abandoned, superseded, or never started

**Archive structure**:
```
planning/archive/
├── 2024-11-broken-implementation/
│   ├── README.md (what failed and why)
│   └── [old planning docs]
└── 2026-01-osdi-restart/
    ├── README.md (what was completed)
    └── IMPLEMENTATION_PLAN.md (when Phase 7 completes)
```

---

## Integration with CLAUDE.md

This planning convention is documented in `CLAUDE.md` to ensure:
- New plans go in `planning/` not `docs/`
- Plans are reviewed and cleaned regularly
- Claude knows to check here for active work guidance

---

## Current Focus

**Phase 1: OSDI ctypes Interface**

See `IMPLEMENTATION_PLAN.md` for details.
