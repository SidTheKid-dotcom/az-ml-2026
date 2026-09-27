# Review: AvinashMalladi/amazon-ml-challenge-2026-entity-resolution (27 Sep 2026)

Repo: https://github.com/AvinashMalladi/amazon-ml-challenge-2026-entity-resolution

**Method.** Read-only through GitHub raw views and the REST API: repo, commit, tree, user and branch metadata, plus every
non-empty text file. Nothing was cloned, downloaded or run. The 884 KB LightGBM model file and the 7 `.pyc` files were
not opened. No code was copied. The measurements in §4 re-implement the repo's *ideas* from scratch as `awk` scripts. They
ran on our own local data and on our best public file (`/home/sidvish/Downloads/u1-k-routes-frvote/matching_results.tsv`,
SHA256 `38cbd04a…`, public 0.981022).

**Verdict.** This is not malicious bait: it contains no prompt injection, malware, network code, test labels or credential
harvesting. It is an **overclaimed**, probably AI-agent-assisted student repo, published mid-competition. The
">0.98"/">0.99" headline has no leaderboard evidence. The validation behind it comes from truth-injected sampled pools, and
one version used training S1s. One claimed score was raised in a docs-only commit. The method is a 14-feature rapidfuzz +
LightGBM pipeline on ~16 inverted-index candidates, the same class as our retired v1 tree. **Nothing in it is a big win for
us.** Two ideas (a "sole-occupant" address anchor and domain-style names) point at real generator signals. When measured
against our best file, the remaining headroom for each is about +0.0001–0.0002 public at most.

---

## 1. Authenticity check

| Item | Finding |
|---|---|
| Created / last push | 2026-09-26 16:28 UTC / 2026-09-27 01:57 UTC (during the hackathon) |
| Commits | 6, one author (`AvinashMalladi`, noreply email), all unsigned; 1 branch, no releases, no tags |
| Stars / forks / licence | 0 / 0 / **none** (all rights reserved: we may not reuse the code) |
| Owner | Account since Aug 2024; 21 public repos of student projects (labs, clones); 2 followers. A second repo, `bnp-unified-customer-360` (MDM hackathon), was created the same day |
| Size | 1.3 MB: 91 tree entries, **37 zero-byte "script" placeholders**, duplicate `src/` and `code/.../src/` copies |
| Score evidence | **None.** No logs (`*.log` git-ignored), no leaderboard screenshot, no public score anywhere |

**Commit timeline.** Each row shows what the commit actually changed.

| UTC | Message (abridged) | What changed |
|---|---|---|
| 26 Sep 16:30 | "complete solution … (F_0.5 > 0.98)" | Docs and a model file. `train_model.py` was **empty**; the doc said validation 0.9842, team "Enigma ER" (Avinash Malladi, Gopalakrishnan) |
| 26 Sep 17:07 | "docs: finalize … F_0.5 > 0.99 analysis" | **Docs only.** Team renamed "CodeCanvas" ("Malladi Avanish (+3 Players)"). **Macro F0.5 0.9842 → "0.9908+"**, P 99.08 → 99.2%, R 95.87 → 96.5%, singleton accuracy 96.84 → 97.4%. "on hold-out validation" was deleted. No code or model change |
| 26 Sep 17:35 | "complete inference, verified with official validator PASS … 1.73M" | Added only two `.pyc` files |
| 26 Sep 17:35 | "chore: update .gitignore" | Emptied `.gitignore` (it had listed `.system_generated/`) |
| 26 Sep 19:08 | "complete production inference …" | Real code: blocking/eval scripts and a 119-line `train_model.py` |
| 27 Sep 01:57 | "… global target assignment and sole occupant anchoring" | Rewrote features/inference/training and **retrained the model**. Docs and `config.json` were **not** updated, so they describe the first-commit model (different feature list, τ 0.72–0.75, 300 trees/63 leaves). The code uses a new 14-feature set, a hard-coded 0.70 threshold and 250 rounds/31 leaves |

**Why the claimed scores are not credible**
- **Sampled pool with truth injection.** Every evaluation script builds its pool from the *first* 50k–200k rows of S2 and
  S3 plus all true targets of the sampled S1s. That is about 1–4% of the 10,320,219-record pool. We measured this trap on our
  own data: a sampled pool gave ~0.98, and the same businesses fell to ~0.83 on the full pool.
- **In-sample.** `debug_portal_score.py` ("Simulating Full Portal Evaluation") scores the first 10,000 ground-truth rows. The
  final `train_model.py` trains on the first 12,000 rows of the same file, so the scored set is **10,000/10,000 training
  S1s** (verified locally by ID). The 2,000-row scripts are mostly out of sample (14/2,000 overlap) but still use the
  sampled pool.
- **Not reproducible.** Several scripts call obsolete function signatures or import from zero-byte files
  (`train_and_validate_lgbm.py`, `test_precision_pipeline.py`).
- The "100.00% empirical recall ceiling" and "99.80% of 18,242 true pairs" come from the same sampled, truth-injected setup.
  Our full-pool ceilings are 0.984–0.993 link recall.
- The prior survey of about 480 public repos
  (`research_notes/Entity resolution missing big wins/competition_public_intel.md`) found no team at ≥0.990 publishing.
  This repo does not change that.
- **Inference, not measured:** this pipeline is in the same class as our v1 tree (about 0.957 on the full pool), so its
  real public score is probably well below our 0.981.

**Data facts are accurate** (checked on our local copy), so the author had the real data:
- 12,527,040 train records and 11,702,133 test records;
- 7,638,365 ground-truth links;
- 123,247 singletons (5.58%);
- a test pool of 9,969,589 records;
- 0 cross-country links (our manual agrees);
- India S1 is 99.94% ASCII (554 non-ASCII rows of 883,188).

**Safety scan**
- **Clean:**
  - No text addressed to AI or LLM agents, no HTML comments, hidden blocks or zero-width characters in the README or docs.
  - No URLs, `curl`/`wget`, base64 or obfuscation.
  - No pickles and no network calls.
  - No test labels and no prediction files; `output/` is git-ignored.
- **Hazards (do not run):**
  - `scripts/autonomous_master_run.py` runs inference, the validator and a zip step, then `git add .`, `git commit` and
    `git push origin main`.
  - Seven `cpython-314` `.pyc` files are committed. Their sizes differ from the committed sources, so they were compiled
    from other source versions and cannot be reviewed.
- **Signs of an AI agent (inference):** the "autonomous" runner, `.system_generated/` in the original `.gitignore`,
  marketing-style docs ("extraordinary", "Zero-Loss"), and the placeholder scripts.

## 2. Method summary (ideas only)

- **Blocking.** Hard country partition, with per-country inverted indexes built from `array('I')` posting lists. Each S1
  collects weighted votes from six channels, and the top 16 candidates are kept (the docs say ~10.7):

  | Channel | Weight |
  |---|---|
  | 4 rarest name tokens (by document frequency) | +20, or +15 if the posting list has >250 entries |
  | First 8 characters of the compact name | +40 |
  | Address numbers of ≥2 digits | +15 |
  | Signature "first number _ street token" | +30 |
  | 3 rarest address tokens | +12 |
  | Exact compact address, when the S1 address is unique among S1s | +50 |

  Postings with >250 entries are skipped (or capped).
- **Normalization.**
  - `anyascii` for Indic scripts and accents.
  - Strips domain TLDs, `dba` and `formerly:`.
  - Drops legal and generic suffixes (including `services`, `solutions`, `group`, `center`) and French forms.
  - Adds hand-listed Indic-transliterated legal tokens: `praivet`, `limitted`, `elelpi`, `emtrpraijej`, `solyusms`,
    `holldingg`.
  - Strips leading zeros from numbers.
- **Model.** LightGBM binary classifier (MIT licence; 250 rounds, 31 leaves) trained on the first 12,000 ground-truth S1s
  against the sampled pool, with no validation split. There are 14 features:
  - name: token Jaccard; rapidfuzz ratio, token_sort and token_set; compact-name ratio; compact containment; first-token
    equality (≥4 characters);
  - address: token Jaccard, ratio and token_sort;
  - numbers: number match and number conflict;
  - flags: both addresses present; "exact address and sole occupant".
- **Decision (final code).**
  1. Drop a candidate if the numbers conflict and name token_sort < 0.88.
  2. **Force-accept** a sole-occupant exact-address match (score 10 + p), ignoring the name.
  3. Otherwise accept if p ≥ 0.70.
  4. Assign one owner greedily, **in batch order** (10,000 S1s per batch; targets claimed by earlier batches are lost to
     later ones).

  No list cap of 11 is enforced (up to 16 matches are possible).
- **France.** Zero-shot: the same US/India model and threshold, plus accent folding and French legal and street
  vocabulary. No French validation, pseudo-labels or probes.
- **India.** `anyascii` transliteration plus the transliterated legal-token list. `GENERIC_FIRST_TOKENS` (sri, shri,
  krishna, balaji, …) is defined but unused.
- **Test data.** Only unlabeled test-S1 statistics: S1 address uniqueness for the anchor, and the greedy one-owner step. No
  pseudo-labelling, no LLM and no embeddings.
- **Scores.** Validation 0.9842 was edited to "0.9908+" with no code change (see §1). The leaderboard score is unknown. **No
  change is linked to any score jump.**

## 3. Rules risks if we copied anything

- **Code: do not copy.** There is no licence, and the code was published mid-competition, so copying invites a
  code-similarity or plagiarism flag in top-team review. Ideas only, with attribution.
- **Model file: never use.** It is another team's weights, trained on unknown samples.
- **Evaluation protocol: do not adopt.** Injecting truth into a sampled pool breaks our guardrail and inflates scores.
- **Transductive test-S1 statistics** (address uniqueness among test S1s): the same category as our existing test-side name
  multiplicity and one-owner step. Low risk.
- **No external data, lookups or test labels are involved.** Libraries: `anyascii` (ISC), `rapidfuzz` (MIT), LightGBM
  (MIT), which we already use.

## 4. Ideas worth testing, ranked (with repo evidence vs our measurement)

Break-even for **adding** a link is P(true) ≈ 0.73–0.79 (manual §4.2).

**1. France/India "single brand token at a unique S1 address" adds — useful but small; bundle only.**
- *Repo evidence:* claimed only. The "sole-occupant anchor" came in the final commit, with no score.
- *Ours, train, measured:* the full rule (S1 compact address ≥15 characters, unique among S1s in the country, pool record
  with the identical compact address) yields 600,285 pairs at precision **0.983**.
  - When the names share ≥1 non-legal token: US 0.9925, India 0.9963. We already accept these.
  - When the names share no token, precision by record-name class:

    | Record-name class | US | India |
    |---|---|---|
    | Single invented brand token (≥7 letters) | **0.974** (11,051 pairs) | **0.986** (14,051 pairs) |
    | Multi-token name | 0.838 (22,742) | 0.953 (54,966; inflated by native script) |
    | Decoy word (holding, participations, développement, …) | **0.748** (779) | **0.785** (633) |

- *Ours, test, label-free, best file.* Status of zero-overlap pairs:

  | Country | Accepted | Given to another S1 | Unowned |
  |---|---|---|---|
  | US | 88% | 7% | 5% |
  | India | 93% | 3% | 4% |
  | **France** | **42%** | 10% | **48%** (11,307 pairs) |

  By class, France leaves unowned 1,830 single-brand pairs (36%), 5,943 multi-token pairs (40%) and 4,760 decoy-word pairs
  (97%). India leaves 1,126 single-brand pairs unowned. Of the 11,307 France pairs, 46% belong to S1s that already hold
  3–4 links and 5% to S1s with empty lists.
- *Value if the US/India precision transfers to France:*
  - single-brand adds: about +100 (France) and +65 (India) entity-points, **≈ +0.0001 public in total**;
  - France multi-token pairs (≈0.84) sit near break-even;
  - decoy-word pairs are below break-even (keep rejecting them).
- *Verdict:*
  - **Already have** in US/India: `forensic_features.s1_multiplicities` has `address_count`, and our acceptance rates match
    the train precision.
  - **Risky as a forced accept:** the decoy and multi-token classes are at or below break-even, and our exact-key hard
    rules already failed (moves 0.014, unowned adds 0.276).
  - The only candidate is a France/India single-brand-token add, bundled into a France upload. It is not worth its own slot.

**2. Domain-style glued names (e.g. "Roo Royal Centers LLC" → "roocentersroyal.com") — mostly already have; low priority.**
- *Repo evidence:* one example, which is real. It is a true link in train (S1-659175866 → S2-677006055). The repo's code
  matches only in-order concatenations (8-character prefix and containment).
- *Ours, train, measured:* single-token `name.tld` records are 269,194 of the pool (2.6%), and **97.5% of them are true
  links**, against a 74.0% base rate. For the owned ones, the glued name relates to the S1 name as follows:

  | Relation to S1 name | Share |
  |---|---|
  | In-order concatenation of the core tokens | 68.4% |
  | Permuted tokens | 3.3% |
  | Near-anagram (≤2 characters off) | 4.0% |
  | Other: initials plus words, or token subsets (e.g. `hbryant.com`, `tlservices.com`, `technicalprime.com`) | 24.2% |

- *Ours, test:* our best file owns 95.6% of the 210,250 domain-style records (US 96.9%, India 95.0%, France 93.6%; France
  shows accent-deleted forms such as `collgejean.com`). Headroom against the 97.5% prior is about 4k links, **≤ +0.00015
  public** even if every one is assigned correctly.
- *Verdict:* **already have** most of it (the cross-encoder sees raw text; `edit_ops.py` has a domain flag and an
  in-order two-token `join`; the plan lists domain forms, including accent deletion). A permutation- and initials-aware
  glued-token feature is the only new piece, and it is not worth a retrain on its own.

**3. Four extra Indic-transliterated legal tokens — useful but minor, label-free.** `limitted`, `emtrpraijej`
(enterprises), `solyusms` (solutions) and `holldingg` (holding) are missing from our `string_gbdt.TRANSLIT` (we have
praivet, praivhet, prayvet, elelpi, limiteda, kampani, inkarporeted). This affects only the string-GBDT fields for
native-script India names. Unmeasured: `anyascii` is not installed locally, so check token frequencies on EC2 first. Add it
to the next vocabulary refresh; do not run it as its own experiment.

**4. Number-conflict hard gate** (≥2-digit numbers disjoint and name token_sort < 0.88 → reject). **Already have** as
features (postcode, unit and alpha-suffix conflicts). As a hard rule it is **risky**: our digit vetoes removed more true than
false links, and French street-disjoint links are mostly one-character typos of true links.

**5. Greedy one-owner assignment.** **Already have** (global, score-ordered, +0.008 public). Theirs depends on batch order
and is worse.

**6. Multi-channel inverted-index blocking.** **Already have** (dense E5 K50 plus lexical/phonetic/key union; the C7
address branch added only 43 true links in 58,911 pairs). Their recall numbers are not comparable (truth-injected pools).

**7. DBA/"formerly:" split, country partition, leading-zero numbers.** **Already have** (`ALIAS_MARKERS`, collective
`BRIDGE`, country blocking, `lstrip("0")` keys).

**8. `GENERIC_FIRST_TOKENS` and first-token-equality feature.** **Not applicable.** They are unused in the repo; our
IDF/multiplicity and HONOR/DECOY vocabularies cover this.

**Do not adopt:** the forced sole-occupant accept, the hard number gate, sampled-pool or in-sample evaluation, and threshold
tuning on a truth-injected pool.

## 5. How the §4 numbers were measured (reproducible, label use = TRAIN only)

- **Compact address:** `tolower` in the C locale, then keep only `[a-z0-9]`, length ≥15, keyed by country. "Unique" means
  exactly one S1 in that country has the key.
- **Name overlap:** at least one shared token of ≥2 characters, after dropping English and French legal forms and articles.
- **Record-name classes:**
  - decoy word: `particip|holding|group(e)|veloppement|development|distribution|international|industries|ventures|overseas|exports|infratech|global`;
  - single brand token: exactly one alphabetic non-legal token of ≥7 letters;
  - multi-token: everything else.
- **Domain-style:** the name matches `^[A-Za-z0-9]+\.(com|net|org|in|fr|co|biz|info|io)$`.
- **Tokenizer caveat:** it is ASCII-only, with no `anyascii`, so native-script India names read as zero-overlap.
- **Test side:** a pair's status is read from our best file: owner equals the S1 ("accepted"), another S1, or absent
  ("unowned"). This step uses no labels.
- **Pricing:** k≈3, one true link added ≈ +0.0625, one wrong link added ≈ −0.21; 1,733 entity-points ≈ 0.001 public.
- The awk scripts are in the session scratchpad (`sole_occ.awk`, `sole_cls.awk`, `classify_domain.awk`; tmpfs, not kept).

## Sources

- Repo and README: https://github.com/AvinashMalladi/amazon-ml-challenge-2026-entity-resolution
- Raw files: `README.md`, `Documentation_template.md`, `models/config.json`, `src/*.py`, and every non-empty `scripts/*.py`,
  under https://raw.githubusercontent.com/AvinashMalladi/amazon-ml-challenge-2026-entity-resolution/main/
- API: `/repos/AvinashMalladi/amazon-ml-challenge-2026-entity-resolution` (metadata), `/commits`, `/commits/{sha}` for all 6
  commits, `/git/trees/main?recursive=1`, `/branches`, `/releases`, `/users/AvinashMalladi`, `/users/AvinashMalladi/repos`
