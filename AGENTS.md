# Engineering instructions

Read PROJECT_STATUS.md, IMPLEMENTATION_PLAN.md and the relevant sections of ARCHITECTURE.md and DECISIONS.md before changing a component. Confirm the current branch and working tree.

## Assignment

For architecture and escalation, use Astra Medium. Assign implementation to Sol Medium and routine verification and documentation to GPT-5.6 Luna Low. Use Terra Low for source research and Terra Medium for visual review. The named configurations in .codex/agents define all seven roles. Choose the smallest available model that can meet the acceptance test. Consider task complexity, specialization, repetition, risk, deterministic tests, cost and prior success. Use Low or Medium for every turn. If a task exceeds Medium, decompose it.

Use a named role in a trusted project session when the client exposes custom agents. Give each worker a file boundary, expected output and a deterministic acceptance check. The project allows three concurrent workers. Use the actual model choices exposed by the client. The desktop spawn menu and CLI model catalog can differ. Check the local run record's turn_context.model and effort before reporting a role as executed. Configuration files alone establish intent.

For a manual handoff, pass the task in a fresh CLI session from the repository root. For example, run the following PowerShell command with the repository's existing environment active.

```powershell
codex exec --strict-config -m gpt-5.6-luna -c 'model_reasoning_effort="low"' 'Read AGENTS.md and PROJECT_STATUS.md. Run pytest and report failures and exact commands. Do not edit files.'
```

Give the next worker the commit, file list, test output, ownership and open questions. If three different fixes fail, or leakage, missing source data, architecture or CUDA compatibility invalidate an assumption, escalate to the troubleshooter. Include milestone, intended behavior, actual behavior, exact errors, affected files, attempts, commit, tests and proposed next decision. Return ordinary implementation to the engineer after the issue is resolved.

## Evidence and version control

Use real data for published results. Identify each source, date, population and missing field. Distinguish measured labels from estimates, experimental video features and unsupported targets. Fit all preprocessing and model parameters on earlier games. Separate retrospective replays from real pregame forecasts. Store adaptive events separately. Keep raw footage, source downloads, credentials and checkpoints outside Git.

Use feature branches for major changes. Test each milestone before merging to main. Record authentic commits when work passes its checks. Keep a release tag tied to the tested revision. Report exact metrics, failed checks and unverified features in PROJECT_STATUS.md. Refer to the latest commit with git rev-parse HEAD. Never backdate commits.

## Graphify

When the user types /graphify, consult the installed graphify skill at ~/.Codex/skills/graphify/SKILL.md before taking other action.
