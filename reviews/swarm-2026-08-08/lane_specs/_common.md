# Swarm Review — Common Lane Instructions

**Project:** FluidSim Warp — a Blender 5.1 add-on porting FluidSim 1.0.226 (FLIP fluid simulation) to NVIDIA Warp/CUDA.
**ROOT (absolute, forward slashes):** C:/Users/Tyler/Downloads/liquid_warp_blender51_port/liquid_warp_blender51_port
**Mode:** STATIC review only. Blender/Warp/GPU are NOT installed. Do NOT run tests or builds.

## Hard Rules
- **READ-ONLY.** Do NOT modify/create/delete any file EXCEPT your findings JSON.
- Cite every finding with exact file path + line number, defensible from specific code. No speculation.
- Report real defects, not exhaustive style nits. Prioritize issues you can defend.
- Use Glob/Grep to map your scope, then Read the relevant regions.

## Severity Calibration
| Severity | Meaning |
|----------|---------|
| critical | Crash, silent data corruption, data loss, broken contract, fail-open safety path |
| high | Significant functionality impact (leak, missing validation, broken flow, numerical error) |
| medium | Partial impact, workaround exists (missing edge case, unclear error) |
| low | Minimal/cosmetic (minor optimization, naming, clarity) |

## Findings JSON Schema
Write a JSON object to your lane's output path (given in your lane spec):
```json
{
  "dimension": "<your dimension name>",
  "agent_type": "general-purpose",
  "files_reviewed": <int>,
  "findings": [
    {
      "id": "<PREFIX>-001",
      "file": "relative/path.py",
      "line": 123,
      "severity": "critical",
      "title": "Short title",
      "description": "Detailed description citing the code",
      "impact": "What could happen",
      "fix": "Recommended remediation"
    }
  ]
}
```

## Final Message
Plain text only (do NOT paste the full JSON). Report:
1. `files_reviewed` count
2. Finding counts by severity (critical / high / medium / low)
3. Your top 3 findings as one-liners, each with its ID
4. Confirm the JSON file was written
