---
name: debug-job
description: Debug and diagnose issues with DIRAC jobs
version: "1.1"
tags: [dirac, job, debugging, troubleshooting]
tools:
  - get_job
  - search_jobs
  - get_job_status_summary
  - get_job_sandboxes
  - get_job_metadata
  - set_job_statuses
  - reschedule_jobs
---

# Debug a DIRAC Job

## Workflow

1. **Identify the problem job(s)**:
   - If the user provides a job ID, use `get_job` to retrieve details.
   - If the user describes a pattern (e.g., "my failed jobs"), use `search_jobs` with appropriate conditions.

2. **Analyze the job status**:
   - Check `Status` and `MinorStatus` for the current state.
   - Check `ApplicationStatus` for error messages.
   - Check `Site` to see where the job ran/is running.
   - Check `LastUpdateTime` to detect stalled jobs.

3. **Inspect sandboxes** (for failed or errored jobs):
   - Use `get_job_sandboxes` to retrieve sandbox download URLs.
   - Check `StdErr` in the output sandbox for error details.
   - Check `StdOut` for application output.

4. **Diagnose based on status**:

### Failed Jobs
| MinorStatus | Likely Cause | Action |
|-------------|-------------|--------|
| `Application Finished With Errors` | Script error | Check StdErr in output sandbox |
| `Input Data Not Available` | Missing input files | Verify input sandbox files exist |
| `Stalled` | Job exceeded wall time or lost heartbeat | Increase `MaxCPUTime` or check site |
| `Exception During Execution` | Runtime crash | Review application logs |
| `Downloading Input Sandbox Failed` | Upload issue | Re-upload input files |

### Stuck/Waiting Jobs
- **Waiting > 24h**: Check if the requested site is available and has free slots.
- **Matched but not Running**: Pilot may have failed at the site. Try a different site.
- **Submitting for too long**: Infrastructure issue. Contact grid support.

### Running Jobs
- Use `LastUpdateTime` to estimate progress.
- If heartbeat is stale (>1h), the job may be stalled.

5. **Take action**:
   - **Kill stalled jobs**: `set_job_statuses(job_ids=[...], status="Killed")`
   - **Reschedule failed jobs**: `reschedule_jobs(job_ids=[...])`
   - For patterns of failures: use `get_job_status_summary` to see the big picture.
   - For failed jobs: suggest fixes and offer to resubmit with corrections.
   - For stuck jobs: suggest alternative sites or parameter adjustments.

## Common Search Patterns

**Find recent failures:**
```
search_jobs(conditions=[
    {"parameter": "Status", "operator": "eq", "value": "Failed"}
], per_page=20)
```

**Find jobs at a specific site:**
```
search_jobs(conditions=[
    {"parameter": "Site", "operator": "eq", "value": "LCG.CERN.ch"}
])
```

**Find stalled jobs:**
```
search_jobs(conditions=[
    {"parameter": "MinorStatus", "operator": "eq", "value": "Stalled"}
])
```
