# Brief A — Generator forensics (Cursor + Grok)

Read `docs/briefs/README.md` first (context, rules, data paths). Your folder:
`/home/sidvish/Documents/Work/Personal/ext-experiments/A/`.

## Question

The data was produced by a synthetic generator. Which **new, near-deterministic
"tells"** separate (a) a true noisy copy from a look-alike, and (b) the true
owner from other businesses that also look like the record? We found one such
tell already: empty-address records are ~98% owned, and the owner is decidable
when the name is unique (+0.004 on the tuning set). Find the next ones.

## Why it matters

Models learn smooth similarity; a generator artifact can be exact. One strong
tell can be worth more than hours of GPU training, and it transfers to France
if the generator is shared.

## Method

1. Build a study set from train labels: sample ~100k S1s (drop HOLDOUT-2 IDs),
   with all their true records, plus look-alike records: unowned records and
   other S1s' records that share the S1's normalized name, or share its full
   digit set, or share its exact address. Use pyarrow projection and filtering;
   stay under 2.5 GB.
2. Reverse-engineer operations. For each (S1, record) pair, align name and
   address token by token and classify the edit operations. Measure how often
   each operation appears in true copies vs look-alikes, per country and per
   source (S2 vs S3). Directions to probe, in order:
   - **How look-alikes are made:** which component changes (unit, house number,
     street word, city, state, name word, legal suffix) and how (±1 on a number?
     a random number? a street taken from another business in the same city?).
     Is every look-alike derived from a specific S1? Does that S1 also have
     true copies?
   - **Vendor fingerprints:** S2 uppercases addresses (63% train); S3 reorders
     components; junk tokens (`##`, `<NULL>`, `>>`). Does a look-alike follow
     its source's style the same way a true copy does? Style mismatches can be
     a tell.
   - **Typo model:** keyboard-adjacent substitution, doubling, deletion, swap.
     Do look-alikes carry typos at the same rate as copies?
   - **Name-only and address-only records:** empty address is done. Are there
     name-empty records, records whose name is a domain or hashtag
     ("#brightathene", "Fs.C0m"), or trade names ("Deltasolzeph")? How often are
     they owned, and by whom can they be decided?
   - **Absent-business clusters:** do unowned records come in groups that look
     like the same (absent) business, and does that distinguish them?
   - **Count structure:** how many copies per S1 in S2 vs S3; does a look-alike
     ever appear in both sources?
3. For every candidate tell, report coverage (share of pairs where it fires)
   and precision (P(true copy | tell), or P(owner | tell)), with n.
4. Test the best tells on v1's tuning scores: `tune_pairs.parquet` (baseline
   accept `original_blend_probability >= 0.775`; macro F0.5 0.968992). Measure
   each tell as a rule (accept/drop), on top of the known name-only rule (do
   not re-report the name-only rule itself), with the entity-level delta.
   The tuning set is reused development data; report it as such.

## Out of scope (other agents own these)

Digit relation classes, same-length substitution counts, street-token Jaccard,
exact name+digits key, name/address multiplicity, French normalization (agent-F);
the name-only rule (agent-D). Build on them; don't rebuild them.

## Deliverable

`report.md`, `results.json`, scripts; plus `tells.py` with pure functions
computing your top tells from `(s1_name, s1_address, rec_name, rec_address,
country, rec_source)`. Rank findings by estimated entity-points on test.
