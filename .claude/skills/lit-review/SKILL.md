---
name: lit-review
description: >
  Search academic literature for the FoG/IMU/TCN project and append curated,
  deduped entries to docs/literature/references.md. Use whenever the human asks
  to "find papers", "check the literature", "is there a better dataset", "what's
  SOTA on X", or to back a DECISIONS.md ADR with citations. Uses the nrp-literature
  MCP (Semantic Scholar + OpenAlex).
---

# Skill: lit-review

## When to invoke
The human wants literature support: new datasets, baselines to beat, methodology
precedent, or citations for an ADR / paper section.

## Tools
- `nrp-literature:search_papers(query, limit, year_from, year_to)` — primary search.
- `nrp-literature:get_paper_detail(doi_or_id)` — only when references/citations of a
  specific paper are needed.
- Do NOT use `export_to_references` (it writes to the unrelated NRP_Claude_Agent
  folder). Write to THIS repo's docs/literature/references.md instead.

## Procedure
1. Turn the request into 1–3 focused queries (3–8 words each). Use `year_from`
   >= 2021 unless the human wants foundational work. Prefer separate queries per
   distinct concept over one broad query.
2. Run the searches. From the returned papers, CURATE — do not dump everything.
   Keep only items relevant to: FoG datasets, wearable-IMU sensor placement, TCN/
   dilated-causal/WaveNet models, detection-vs-prediction methodology, edge
   deployment, or generalization/fairness.
3. For each kept paper, write ONE compact entry in references.md format:
   `**Authors Year** — title. *Venue.* DOI: ... · cites N. [TAG] one-line relevance.`
   - TAG ∈ {DATASET, METHOD, METHODOLOGY, REVIEW}.
   - The relevance line is in our own words and says how it bears on OUR project
     (link to an ADR number when possible).
4. Dedupe by DOI against existing entries. Append under the correct section; create
   a section only if needed.
5. Only record DOIs/venues returned by the tool. NEVER fabricate a citation. If a
   field is missing, write "(n/a)" rather than guessing.

## Output rules
- Keep references.md lean (bibliographic line + 1 relevance line per paper). Long
  per-paper notes, if requested, go to docs/literature/<firstauthor_year>.md.
- Copyright: paraphrase. Do not paste abstracts. At most one short (<15-word) quote
  per paper if exact wording matters; otherwise none.
- After appending, report back: how many papers found, how many kept, and the 1–2
  most decision-relevant findings (R10/R12 in CLAUDE.md). Surface if a query
  returned nothing useful.

## Example
Human: "is there a bigger FoG dataset than Daphnet for prediction?"
→ search ["freezing of gait dataset deep learning prediction", "tDCS FOG DeFOG
  accelerometer benchmark"], year_from 2021 → curate → append [DATASET]/[METHOD]
  entries → report: "DeFOG/tDCS (~60–90 subjects, prediction-framed) is the main
  candidate; O'Day 2022 is a smaller open single-ankle option. See ADR-008b."
