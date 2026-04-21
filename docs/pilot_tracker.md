# Pilot Prep Tracker

## Objective

Prepare a credible pilot case for external compute access by:
- validating the revised VES pipeline locally and on GPU
- finalizing proposal materials
- pursuing Purdue access through a faculty introduction
- preparing fallback compute options if Purdue is not viable

## Technical Validation

- [x] Local smoke test completed successfully on CPU
- [x] Medium local CPU run attempted and timed out at 20 minutes on 2048 samples
- [ ] Receive Hugo's GPU validation results
- [ ] Compare CPU vs GPU runtime and loss behavior
- [ ] Decide whether to clean up the tracked large manifest file

## Repository / Branch

- [x] Push test branch for Hugo
- [x] Send Hugo repo link and exact test request
- [ ] Confirm branch is sufficient for Hugo's environment
- [ ] Decide whether to split docs/proposal work from runnable test branch

## Proposal Materials

- [x] Revised proposal draft created
- [x] Appendix added
- [x] References added with access dates
- [ ] Final proofread of proposal
- [ ] Draft one-page cover letter
- [ ] Create separate long-form pilot evaluation report

## Purdue Outreach

- [ ] Finish soft-introduction email to emeritus faculty contact
- [ ] Send outreach email
- [ ] Identify likely current Purdue contacts
- [ ] Prepare short verbal summary for meetings
- [ ] Prepare faculty-facing Anvil pilot summary

## Purdue Compute Path

- [x] Preliminary platform analysis favors Anvil first, Gilbreth second
- [ ] Prepare a concrete Anvil pilot run plan for presentation
- [ ] Prepare a plain-language explanation of Slurm/job scheduling for personal reference
- [ ] Turn the Anvil plan into a faculty-facing one-paragraph summary

## Fallback Compute Planning

- [ ] Build a list of fallback compute providers
- [ ] Categorize fallback options:
  - academic HPC
  - research cloud credits
  - company-donated compute
  - nonprofit or open-science programs
- [ ] Draft a reusable bounded pilot ask
- [ ] Define the minimum acceptable compute package:
  - 1 modern GPU
  - short pilot duration
  - subset training only

## Communications

- [x] Hugo has received the paper draft and test request
- [ ] Decide when to follow up with Hugo if no timeline emerges
- [ ] Prepare outreach version of the project summary for new contacts
- [ ] Prepare a short explanation of why this is not just another OCR project

## Notes

- Local smoke test succeeded after script and training-path fixes.
- Local 2048-sample CPU run exceeded the 20-minute timeout, which strengthens the case for GPU-based pilot access.
- Hugo has the paper draft, repo link, and test instructions. GPU timing data is still pending.
