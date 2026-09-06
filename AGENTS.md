# Project instructions

At the start of every new session, before analysis, review, or code changes, read and follow both workspace instruction files:

- `../AGENTS.md`
- `../CRITERIA.md`

Resolve these paths relative to this file, not the shell working directory. They apply to every task in this repository and its subdirectories without the user repeating them. Read the current files rather than relying on previous chat memory.

If this repository is opened in a separate checkout or worktree where those files are absent, use `E:/Downloads/USTH_ICTLab/AGENTS.md` and `E:/Downloads/USTH_ICTLab/CRITERIA.md` on this machine. If neither location is accessible, report the missing instructions and ask for their location before continuing project work.

Always start responses with `okw`.

For PAFA/PatchCore experiment work, also read and follow `../PAFA_Experiment_Roadmap.md` and `../PATCHCORE_CODEX_LIGHT_AGENT_RUNBOOK_W2_UPDATED.md` before planning, editing, or running anything. Treat `../PATCHCORE_CODEX_LIGHT_AGENT_RUNBOOK_W2_UPDATED.md` as the current implementation/runbook authority replacing the deleted `../PAFA_PatchCore_Experiment_Plan_v2_docx.md`. Do not run an experiment, training command, evaluation command, or long script until the current experiment stage, required rule files, success criteria, stop criteria, dataset assumptions, and output artifacts are clear. If either the roadmap or the W2 runbook is missing, stop and report the missing file instead of guessing or running.

Usage-limit guardrail: if the 5-hour usage window has 6% or less remaining, stop before starting new work or running commands. Leave a concise handoff containing the current stage, files read, changes made, checks run, unresolved rules/problems, and the next safe command or decision. Continue only after the user resumes with enough usage available.
