---
name: search-jobs
description: Search and filter DIRAC jobs using natural language or structured queries
version: "1.0"
tags: [dirac, job, search, monitoring]
tools:
  - search_jobs
  - get_job_status_summary
  - get_job_metadata
---

# Search DIRAC Jobs

## Workflow

1. **Parse the user's request** into structured search conditions.
2. **Build conditions** using the operator reference below.
3. **Call `search_jobs`** with the conditions and relevant parameters.
4. **Summarize results** in a clear, tabular format.

For a high-level overview, use `get_job_status_summary` first.
For full details on specific jobs, use `get_job_metadata` with the job IDs.

## Operator Reference

### Scalar operators (single value)

| Operator | Meaning | Example |
|----------|---------|---------|
| `eq` | Equal | `{"parameter": "Status", "operator": "eq", "value": "Failed"}` |
| `neq` | Not equal | `{"parameter": "Status", "operator": "neq", "value": "Done"}` |
| `gt` | Greater than | `{"parameter": "JobID", "operator": "gt", "value": "1000"}` |
| `lt` | Less than | `{"parameter": "JobID", "operator": "lt", "value": "2000"}` |
| `like` | SQL LIKE pattern | `{"parameter": "JobName", "operator": "like", "value": "%analysis%"}` |
| `not like` | Negated LIKE | `{"parameter": "JobName", "operator": "not like", "value": "%test%"}` |
| `regex` | Regular expression | `{"parameter": "JobName", "operator": "regex", "value": "^prod_.*"}` |

### Vector operators (multiple values)

| Operator | Meaning | Example |
|----------|---------|---------|
| `in` | In set | `{"parameter": "Status", "operator": "in", "values": ["Failed", "Killed"]}` |
| `not in` | Not in set | `{"parameter": "Site", "operator": "not in", "values": ["LCG.CERN.ch"]}` |

## Common Patterns

**Recent failures:**
```
search_jobs(conditions=[
    {"parameter": "Status", "operator": "eq", "value": "Failed"}
], per_page=20)
```

**Jobs at a specific site:**
```
search_jobs(conditions=[
    {"parameter": "Site", "operator": "eq", "value": "LCG.CERN.ch"}
])
```

**Jobs by owner:**
```
search_jobs(conditions=[
    {"parameter": "Owner", "operator": "eq", "value": "username"}
])
```

**Jobs with multiple statuses:**
```
search_jobs(conditions=[
    {"parameter": "Status", "operator": "in", "values": ["Failed", "Stalled", "Killed"]}
])
```

**Jobs matching a name pattern:**
```
search_jobs(conditions=[
    {"parameter": "JobName", "operator": "like", "value": "%production%"}
])
```

## Searchable Parameters

Common parameters: `JobID`, `Status`, `MinorStatus`, `ApplicationStatus`, `JobGroup`, `Site`, `JobName`, `Owner`, `LastUpdateTime`, `SubmissionTime`.
