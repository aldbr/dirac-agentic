---
name: submit-job
description: Submit a job to the DIRAC grid computing infrastructure
version: "2.0"
tags: [dirac, job, grid, computing]
tools:
  - submit_job
---

# Submit a Job to DIRAC

## Workflow

1. **Gather requirements** from the user:
   - What executable to run
   - Any command-line arguments
   - Input files needed (input sandbox)
   - Output files to retrieve (output sandbox)
   - Site preference (optional)
   - Resource requirements: memory (MB), CPU time (seconds)

2. **Submit the job** using the `submit_job` tool with the gathered parameters:
   ```
   submit_job(
       executable="/path/to/script.sh",
       job_name="Descriptive Job Name",
       arguments="--flag value",
       input_sandbox=["script.sh", "data.txt"],
       output_sandbox=["StdOut", "StdErr", "result.dat"],
       site="LCG.CERN.ch",
       memory=2048,
       max_cpu_time=3600
   )
   ```

3. **Report the result**: Provide the job ID(s) returned on success.

## Common Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `executable` | Script or binary to run | `/bin/echo` |
| `arguments` | CLI arguments | `"hello world"` |
| `input_sandbox` | Files uploaded with the job | `["run.sh"]` |
| `output_sandbox` | Files downloaded after completion | `["StdOut", "StdErr"]` |
| `site` | Target execution site | `LCG.CERN.ch` |
| `memory` | Required memory in MB | `2048` |
| `max_cpu_time` | Max CPU time in seconds | `3600` |

## Examples

**Simple echo job:**
```
submit_job(executable="/bin/echo", arguments="hello world")
```

**Python script with dependencies:**
```
submit_job(
    executable="python",
    arguments="analysis.py --input data.root",
    input_sandbox=["analysis.py", "data.root"],
    output_sandbox=["StdOut", "StdErr", "output.root"],
    memory=4096,
    max_cpu_time=7200
)
```

**Advanced (raw JDL):**
```
submit_job(jdl_content='Executable = "/bin/echo";\nArguments = "hello";')
```
