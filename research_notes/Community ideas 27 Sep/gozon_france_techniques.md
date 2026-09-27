# GozonLegendent (public LB 0.987326): France and decision-layer techniques to reimplement

Written 27 Sep 2026, about 14:05 IST.

## 0. Source availability: read this first

- **The repo is gone.** `github.com/GozonLegendent/amazon-ml` returns 404 on the web UI, the API (`repos/…`, `/commits`) and raw. It was deleted or made private after its last push (06:32 UTC / 12:02 IST).
- **No copy exists anywhere I could reach:**
  - no fork with that parent (checked 15 recent `amazon-ml*` forks);
  - no GitHub code-search or commit-search hit for `fr_swap`, `fr_swap_ext`, `cl_same_qnum`, `v6pf` or `postfix.py`;
  - Software Heritage has no origin;
  - the Wayback Machine is offline ("Temporarily Offline");
  - the Exa and Parallel caches have nothing.
- **Teammates' accounts have no copy.** Neither GozonLegendent's other public repos nor teammate `manankapoor23`'s repos contain it.
- **So the SOURCE CODE (`src/postfix.py`, `decide.py`, `crossencoder.py`, `features2.py`, `textnorm.py`) could not be read.** No code-level line ranges exist.
- **What was read:** our own 12:19 IST sweep had cached the full text of the two docs:
  - `Documentation_template.md` (DOC)
  - `TEAM_HANDOFF.md` (HO)
  - the commit list

  The cache is at `/tmp/claude-1000/-home-sidvish-Documents-Work-Personal-az-ml-new/41da6ea4-f15c-42cd-93af-ae21d9c57893/scratchpad/triage/GozonLegendent__amazon-ml.txt`. It is session scratch and may vanish.
- **How to read the citations.** They give the doc section and the **cache line numbers** (`c.Lx–y`):
  - DOC line = c.L − 1;
  - HO line = c.L − 434.

  Key rule definitions are quoted verbatim below, so this note stands alone.
- **Code identifiers are named in the docs** but not visible: `src/postfix.py` (post-rules), `src/decide.py` (threshold/expected-F0.5), `FINAL_DROP` in `src/ranker.py`, `src/bucket_diag.py` (threshold diagnostic), `src/border_diag.py` and `src/density_diag.py` (France bands), `src/veto.py` ("unused post-filter"). See DOC Appendix A, c.L366–392.
- **Latest commits (UTC):**
  - `f8fcd0edaf` 06:08 "fr_swap_ext rule; final package docs and run.sh for the three-cross-encoder model"
  - `950949519f` 06:09 "team handoff notes"
  - `e5645855d2` 06:32 "shrink: tighter candidate set reusing pair-level scores, with recall/size sweep"

  The shrink commit changes only `candidate_pairs.tsv` size, not the score.
- **No LB result for `fr_swap_ext` / v6pfx was ever published.** The only statement is "expected about 0.9875–0.9877" (HO §0, c.L439). The team had "one leaderboard attempt left" at 11:45 IST.

---

## 1. How French rows are produced (Q1)

**Same model as US/India. There is no France-specific model, threshold or vote.**
- The pipeline runs per country label, and France goes through it unchanged:
  1. e5-small bi-encoder kNN;
  2. LightGBM pruner;
  3. three listwise cross-encoders (CEs);
  4. a 97-feature LightGBM final ranker;
  5. argmax;
  6. t = 0.75.
- DOC §1 c.L16–29; HO §3 c.L461–473.
- "France has no validation, so it uses the same threshold" (DOC §5.3, c.L317).

**France-motivated changes, made globally inside the model (not France-only branches):**
1. **Address structure.**
   - What they found: "S1 [admin component] always a région; S2/S3 same région ~⅓, the département ~⅓, absent ~⅓ … Whole-address similarity for French S2 therefore looks like training S3."
   - Their fix: "We compare the street component and the entity's finest locality instead." DOC §2.1 c.L97–107.
   - The features behind it: "a street-component-only similarity; whether the entity's two finest localities (learned from Source 1 frequencies) appear in the record, and how common the finest locality is" (DOC §4 c.L191–193).
2. **Source indicator dropped.** "We also drop the source indicator from the model, so French S2 records are not judged as out-of-distribution S2." (c.L106–107). `q_src` is computed but excluded: 50 features computed, 49 used (c.L180).
3. **Normalisation.** "legal words, French/US/Indian street types, bis/ter, N°" (HO c.L462–463).
   - Handoff §5 (c.L516–517): "French text normalisation: already handles R./AV./BD, N°, bis/ter, accents, bracketed legal forms, capitals. Only a minor apostrophe gap ('L'X' vs 'L X'), worth ~0.0001."
   - Compliance note: `textnorm.py` encodes "generic spelling conventions only (Rd/Road, Ltd/Limited, R./Rue, N°, bis/ter, dba)" (c.L404–406).
   - The docs do not say whether the bi-encoder sees normalised or raw text.
4. **Core name** = name with "legal suffixes and honorifics removed; alias forms resolved to the real business" (c.L185–186). This is the unit that `fr_swap` compares.

**Decision layer (all countries):**

| Item | Value | Cite |
|---|---|---|
| Record→entity | each S2/S3 record keeps only its argmax S1 | DOC §4 c.L256–257; HO c.L471 |
| Threshold | p ≥ **0.75** on the final-LightGBM probability of that argmax link | c.L24–27, c.L268–271 |
| "No match" | explicit only inside the listwise CEs (learned slot). At decision time "no match" = argmax p < 0.75 | c.L228 |
| Pruner (candidate list) | ≤ **4** candidates per record, p ≥ **0.003** (recall 99.12%) | c.L157–159 |
| Per-S1 cap | remove links beyond **5 S2 or 6 S3** per entity ("the training maximum"), lowest probability first. 121 links in total | c.L252–253 |
| Min list length | none. Empty rows allowed (singletons 5.6%: "empty prediction earns 1.0") | c.L56 |
| Max list length | effectively 11 (5 S2 + 6 S3) via the cap | — |
| Expected-F0.5 top-k | implemented and compared on fold 0 (enumerate labelings with temperature-calibrated p; k = 0 = singleton). **Not used**: the submitted file is a fixed threshold | c.L258–264, c.L266–271 |

**France sanity check (label-free).** "At the same threshold France gets **3.31 links per entity** against **3.39 for US and India** (whose training truth is 3.46)" (DOC §5.5, c.L329–333).

---

## 2. Post-rules (Q2): `src/postfix.py`, removal only

Verbatim definitions are from DOC §4 c.L244–253 and HO §4 c.L497–503.

### 2a. `fr_swap` (France only). Measured: 0.986201 → **0.987326** (+0.001125 public LB)
> "same first house number, exactly one core word replaced by another, both words from a qualifier vocabulary learned from the file's own French links (words removed ≥ 200 times and added 0.2–5× as often, minus words true pairs add or drop as noise). These are sibling businesses at one address ('Regards Ecole' vs 'Regards Amis'); each qualifier is swapped with ~25 different partners at random, unlike a synonym. 8,703 links."

- The A/B was clean: v6nc vs v6pf, same model (DOC §5 table c.L285–286, c.L292–294). It removed 8,703 links.
- With France at ~15% of the board weight, that implies about **+0.0075 French F0.5**.

**What counts as a "qualifier" (our reconstruction; the code was not visible):**
1. Take every **predicted** French link (entity, record) in the output file, i.e. after argmax and t = 0.75, so this is test-internal and label-free.
2. Compare core-name token multisets.
   - A token in the entity but not the record is "removed".
   - A token in the record but not the entity is "added".
3. Count `removed[w]` and `added[w]` over all French links.
4. `w` is a qualifier if **removed[w] ≥ 200** and **0.2 ≤ added[w] / removed[w] ≤ 5**.
   - Rationale: a random swap generator moves a word in both directions about equally.
   - True-pair noise is directional. Records gain "Services", "Fils", "Associés", "France", or lose legal forms.
5. Then subtract a **noise list**: "words true pairs add or drop as noise". Named examples:
   - swaps INTO "Services", "Fils", "Associés", "France";
   - garbled typos;
   - "the same noise exists in US/India true pairs: Clinic → Services, Center" (HO c.L500–501).

   The most likely source is words that US/India **training true pairs** add or drop.
6. Remove a link iff:
   - it is French;
   - the **first house number is equal**;
   - the core names differ by **exactly one word replaced by one word**;
   - **both** words are qualifiers.

   No street-similarity condition is stated.
7. "~25 different partners" is a stated property of qualifiers, not a stated rule condition. Partner diversity ≥ ~10 is a cheap extra guard.

**House-number handling in the rule:** the rule applies only when the first number is **equal**. Numbers that differ are left to the model. "A number mismatch alone is therefore not a reliable reject rule" (DOC §2.1, c.L72–75).

### 2b. `fr_swap_ext` (v6pfx, commit `f8fcd0edaf`). LB never published
> "the record's NEW word is a French qualifier and the entity's word is any other non-noise word ('Palestine Sportive' → 'Palestine Club'), ~3,200 more links removed." (HO c.L502–503)

- So the rule is: same first number, one-for-one core-word swap, `added_word ∈ qualifiers`, and `removed_word ∉ noise`. The removed word need not be a qualifier.
- The team expected 0.9875–0.9877 (+0.0002–0.0004 over 0.987326). This is unmeasured.
- Under their cost rule, the removal pays if ≥ ~26% of the removed links are wrong.

### 2c. `namesake` (all countries)
> "no address, and ≥ 2 Source 1 entities in the country share the exact name. 1,366 links."

- It was introduced together with the e5-base CE in v5pf (0.986306 → 0.986451), so it is **not isolated**.
- Motivation: 75% of their validation loss is address-less records with shared names. Following another source's identical-name record "picks the right entity 19.6% vs 21.9% at random" (c.L320–325).

### 2d. `cap`
- More than 5 S2 or more than 6 S3 links per entity: the lowest-probability extras are removed. 121 links. Not isolated.

### 2e. Tried and not used
- `src/veto.py` is an "unused post-filter". Its content is unknown.
- Lowering the France threshold was rejected. "France's uncertain bands are word swaps, moved-number siblings and no-address records, not formatting noise. Not safe." (HO c.L510–511)

### Mapping to our `ber/k_frswap.py`
Our rule removed 4,458 links; theirs removed 8,703 (+~3,200 ext). Ours is stricter in three ways:
- **Vocabulary.** Ours: both words in ≥ 20 French S1 names. Theirs: the swap-symmetry vocabulary learned from the links.
- **Street.** Ours also requires street `fuzz.ratio ≥ 85`. They state only "same first house number".
- **Word pair.** Ours has a typo guard (difflib < 0.6) and an abbreviation guard. They only subtract a noise list.

The gap is where the likely gain is.

---

## 3. The 0.75 threshold, label-free (Q3). Applied to ALL countries, not France-specific

Source: DOC §5.3, c.L309–319; HO §4 c.L493–496; `src/bucket_diag.py`.

1. For US/India (same generator as train), take fold-0 validation **true links per entity per probability band**.
2. Divide by **test links per entity** in the same band and country. This gives "implied test precision". Their result:
   - **40–54% below p = 0.75**;
   - **73–94% above** it.
3. Break-even under macro F0.5 is about 77% precision ("a false link costs an entity about 0.19, and an extra true link gains about 0.056"). The handoff states it as: add pays at ≥ ~74% precision; remove pays at ≥ ~26% wrong.
   - My check with 3.46 true links per entity: one false link costs 0.188 and one missed link costs 0.075.
4. So t = 0.75. This costs −0.0008 on validation (0.99305 at t = 0.36 → 0.9922).
5. On the LB, the 0.969 → 0.986306 jump is v3. It is **confounded** with dropping the four cluster-count features.
6. France reuses t = 0.75. Its only check is the link density (3.31 vs 3.39).

**Rejected diagnostics (do not repeat):**
- Matching link-count histograms "always prefers the lowest threshold".
- Rule-matched "silver" pairs "agree ~1.0 in every band even where true precision is 0.77" (HO §5, c.L512–515; `src/silver_bias.py`).

**Mapping.** Their test had "about 2× more links in p 0.36–0.75" than validation. Our 26 Sep diagnostic found the opposite for our pair tree: test US/India borderline densities matched B_REST. So the 0.75 shift may not transfer one-for-one. It is cheap to redo the band ratio on K at 0.75 vs lower values.

---

## 4. House-number features (Q4)

**Decoy features (`features2.py`, on pruned pairs; 29 computed, 26 used)** (DOC §4 c.L209–212):
- first-number equality, **containment**, gap and **signed gap**;
- set-level:
  - "a number moved to a nearby value (with its signed distance)";
  - shared numbers;
  - the share of each side's numbers found on the other side.

**Blocking-stage features** (c.L194): number overlap (TF-IDF, shared count, Jaccard, first-number equality).

**Not mentioned:** digit edit distance, ratio, explicit truncation/padding flags. "Containment" is the likely stand-in for a dropped digit (924 ⊂ 4924) and zero-padding (00955 vs 955).

**Why these features** (training labels, US; c.L66–75):

| Signal | Decoys | True pairs |
|---|---|---|
| First number equal | 0.9% | 77% |
| Moved to a nearby value | ~90% | 0.5–1.7% |

True-pair number noise includes zero-padding `00955`, a dropped digit `924`/`4924`, prefixes `B3/303`, and India `H.NO 53` injection (14%).

**France:**
- There is no France-specific number handling (bis/ter is normalised).
- Nothing is isolated for France.
- Their French uncertain bands are "word swaps and other name changes at the **same** house number" (c.L330–333), plus "moved-number siblings" (HO c.L510).

**Relevance to our 0.2 vs 0.036 per S1 gap:**
- Their France density (3.31, 98% of US/India) suggests that they, unlike us, keep French same-name different-number true links.
- The model decides these through the containment and signed-gap features, not a rule.
- A French add-rule restricted to **containment/padding/prefix** number relations (never a substitution to a nearby value) fits their evidence. It must reach ≥ 74% precision.

---

## 5. Listwise cross-encoder (Q5)

Source: DOC §4 c.L226–241; HO c.L467–468.
- **Input:** `<S1 text> </s> <record text>`. The field layout inside each text (e.g. `name | address`) is not documented.
- **Head:** mean-pooled, linear head in fp32, encoder in bf16. One logit per (S1, record).
- **Softmax direction:** "listwise softmax over each **record's** candidates plus a learned 'no match' slot", i.e. **per record over candidate S1s** (the assignment direction), not per S1.
- **Candidates per list:** the pruner's ≤ 4, so a list has at most 4 + 1 slots.
- **The no-match slot:** "learned". The most plausible implementation is a trainable scalar logit appended to each record's list, with target = null for the ~26% unmatched records. The code was not visible, so a record-conditioned null head cannot be ruled out.
- **The three CEs** (all trained on folds 5–9):

  | CE | Base model | Initialisation | Training queries | Learning rate |
  |---|---|---|---|---|
  | 1 | e5-small | from the fine-tuned bi-encoder | 800k | not stated |
  | 2 | e5-base | — | 1.5M | 3e-5 |
  | 3 | bge-reranker-v2-m3 | — | 800k | 2e-5 |

  Epochs, batch size and max length are not documented.
- **Into the final LightGBM:**
  - each logit, plus its **rank, margin and gap within the record's group AND within the entity's group**;
  - the entity's **count of positive CE logits**;
  - the record's candidate count.

  Top gain features: the bge-reranker margin over the record's runner-up, the e5-base margin, the two logits, the pruner probability.
- **LB ladder:** each extra CE added +0.0001 to +0.0009.

---

## 6. Other plausible reasons for their France advantage (Q6)

1. **They dropped density/cluster features** (`cl_same_qnum, cl_same_snum, cl_n, pr_s_n`).
   - With these features, test decoy-signature acceptance was 9.1% vs 0.07% on validation. Without them it fell to 0.98%.
   - This was worth most of 0.969 → 0.986 (c.L296–308).
   - Our K uses claimant-set features, and our adversarial AUC was 0.78–0.83. France is test-only, so any density shift hits it unmeasured. FINAL_A (C2-free K) should be checked on the same signature.
2. **One structurally robust model** for France: street component, finest locality, no `q_src`. We use a cross-family vote instead. Replicating this means retraining (> 2 h).
3. **The fold protocol assigns each decoy to the fold of the entity it competes with**, so validation false merges are not understated (c.L120–124). This matters for threshold honesty, not for France directly.
4. **`namesake` applies to all countries.** It uses the exact raw name and a ≥ 2-S1 count within the country.
5. **The qualifier vocabulary is learned from the prediction file itself.** It adapts to the French generator vocabulary without labels, where our list is based on S1 frequency.

---

## 7. Ranked for us: expected France gain per hour

| # | Action (exact parameters) | Expected gain | Time |
|---|---|---|---|
| 1 | **Re-derive `fr_swap` their way**: removed ≥ 200, 0.2 ≤ added/removed ≤ 5 over our own French links (core names); subtract US/India true-pair one-for-one swap noise words and `services, fils, associés, france`; same first house number; one-for-one core-word swap; drop our street ≥ 85 and S1-freq ≥ 20 gates (keep them as a variant). Diff against our 4,458 removals. | Their 8,703 links = +0.0011 LB. Our extra ~4k at similar precision: maybe +0.0003–0.0006 LB (+0.002–0.004 France) | 45 min |
| 2 | **`fr_swap_ext`**: added word ∈ qualifiers, removed word ∉ noise | Their ~3,200 links, expected +0.0002–0.0004, **unmeasured** | +15 min |
| 3 | **France density check**: our links/S1 in France vs US/India at the same τ (theirs: 3.31 vs 3.39). If ours is ≪ 3.3, test a containment/padding/prefix-number add for France, never a nearby substitution. Must pass ≥ 74% precision on US/India B_REST first | Recall side of the 0.2 vs 0.036/S1 gap | 45 min |
| 4 | Band implied-precision diagnostic on K / FINAL_A (τ 0.60–0.80) | Threshold check | 30 min |

Already matching theirs: argmax per record, S2 ≤ 5 / S3 ≤ 6 cap, namesake (verify: exact name, ≥ 2 S1, no address).
