# AI use log

Tool: Claude Sonnet 5.5 (Anthropic), used through Claude Code and chat. Date of work: 24 October 2026.

## What AI was used for
- Repository structure, pipeline code (loading, time grid, features, models, validation), unit tests.
- First drafts of the report sections.
- Step-by-step guidance on Git and GitHub.

## Errors in the AI's work found during checking
- First plan capped gap-filling at 3 hours. Corrected to a 72-hour rule based on the gap distribution.
- Assumed the environmental data could be linked to rooms. No shared key exists in the supplied files, so it is out of scope.
- First training-size test used sizes larger than the available training data. Replaced with 10k, 50k, 100k and 200k rows.
- Described the held-out test as scoring two rooms. Only one room (cca1ac3e) had validation and test rows.
- Described the missing data as scattered gaps. It is mostly one room (3e5a14df).
- Described the model as using 200 boosting iterations. 200 is the maximum, and 113 were fitted.

## Decisions
- Acceptance margin set to 0.03 F1 after viewing validation results (gain 0.045) and before scoring the test period. The planned margin of 0.05 was dropped.
- Test period scored once. No code was changed after seeing test results.
