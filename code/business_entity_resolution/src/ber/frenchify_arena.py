"""Train-derived French-operation arena for C2-FR (option A: no test text in training data).

``build`` writes an arena that ``ber.fast_trainer --command train --mode c2`` loads unchanged
(same layout as ``ber.france_mix_arena``: manifest, reference/pool/pairs, v1_blend/v1_tree,
labels dir, integer partition 0=A, 1=B_REST, 2=SLICE).

Parts
  * FR (partition A): a seeded sample of arena partition-A S1s (US and India) with their real
    true records and real candidate decoys (TRAIN labels), rewritten by French operations:
      - consistent base map: English/Indian legal forms -> French forms; street types ->
        French street types placed before the street name (with an article chosen per street);
        the last admin component (US state / Indian state) -> a French region token;
        country -> France;
      - per-record French style operations applied to every record (true or decoy) at the
        same rates, so no style op is a label shortcut: legal form dropped / added / dotted,
        trailing copy suffix (Cie/Services/Associés/et Fils), street abbreviation, number
        format (N°/No/#/(n)/zero pad/bis/ter), region -> département / dropped, upper -> title
        case, accent dropping, "(France)" bracket kept/dropped;
      - synthetic hard negatives: legal-form swap + house number + delta, decoy word + number
        shift, sibling S1s (same core name + number + city, other street, other legal form)
        with their own synthetic copies (positives of the sibling, negatives of the original).
    Real pairs keep the arena v1 tree/blend of the original pair (a proxy: not recomputed on
    the French text). Synthetic negatives get tree = blend = 0.5 (in the tree gate; typed as
    lookalike / contest-loser negatives by fast_trainer).
  * TU (partition SLICE): a disjoint Frenchified tuning world built the same way from other
    partition-A S1s (labels by construction); never trained on.
  * Replay (A): seeded partition-A S1s, all candidate pairs, original text and TRAIN labels.
  * Monitor (B_REST): the 20,000 B_REST S1s of the 26 Sep france_mix monitor, all pairs.
  * Owner stubs (A): owners of records in the mixed pool that were not sampled; no pairs.

Vocabulary tokens are used only if they occur in the supplied French rows (``vocab.json``
from ``scripts/vocab_probe.py`` counts); zero-count tokens are dropped and logged.
No test text enters the training data. No HOLDOUT-2 label or SLICE label is read.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import unicodedata
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber.v2_ce_train import file_sha256, sealed_ids

CODE_A, CODE_B_REST, CODE_SLICE = np.uint8(0), np.uint8(1), np.uint8(2)
TEXT_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
CHUNK = 8_000_000
DELTAS = (1, 3, 5, 7, 9, 11, 13, 21)  # generator decoy offsets measured on train
EVEN_DELTAS = (2, 4, 6, 8, 10, 12)  # even shifts seen in French data (lead, 27 Sep: e.g. 48 -> 50); half the draws
SYN_SCORE = np.float32(.5)  # synthetic-pair v1 tree proxy (in the tree gate)
# pair kinds (int8) recorded for audit
K_REAL_POS, K_REAL_NEG, K_LEGAL_SWAP, K_DECOY_WORD, K_SIB_POS, K_SIB_VS_ORIG, K_ORIG_VS_SIB, K_LEGAL_POS = range(8)
KIND_NAMES = ["real_positive", "real_decoy", "syn_legal_swap_delta", "syn_decoy_word_shift",
              "syn_sibling_own_copy", "syn_sibling_vs_original_records", "syn_original_vs_sibling_copies",
              "syn_same_number_legal_added_or_dropped_copy"]


def draw_delta(rng) -> int:
    return rng.choice(DELTAS) if rng.random() < .5 else rng.choice(EVEN_DELTAS)


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def _json(path: Path, value) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _npy(path: Path, value) -> None:
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, value)
    tmp.replace(path)


def h32(*parts) -> int:
    return zlib.crc32("\x1f".join(str(p) for p in parts).encode("utf-8"))


def fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


# ---------------------------------------------------------------------------
# Vocabulary (candidates; kept only when counted > 0 in the supplied French rows)
# ---------------------------------------------------------------------------

LEGAL_MAP = [  # English/Indian legal phrase (regex, case-insensitive) -> French form
    (r"private\s+limited|pvt\.?\s*ltd\.?|pvt\.?\s+limited|private\s+ltd\.?", "SARL"),
    (r"l\.l\.c\.?|llc", "SARL"),
    (r"incorporated|inc\.?", "SAS"),
    (r"corporation|corp\.?", "SA"),
    (r"limited|ltd\.?", "EURL"),
    (r"l\.l\.p\.?|llp", "SCI"),
    (r"company", "Societe"),
]
FR_FORMS = {"SARL": .283, "SAS": .201, "EURL": .065, "SA": .049, "SASU": .041, "SCI": .032,
            "Association": .015, "Societe": .016}  # generator_params france_label_free S1_forms (+ Societe, S1 count)
DOTTED = {"SARL": ["S.A.R.L.", "Sarl"], "SAS": ["S.A.S.", "S.A.S", "Sas"], "EURL": ["E.U.R.L.", "Eurl"],
          "SA": ["S.A."], "SASU": ["Sasu"], "SCI": ["Sci"], "Association": ["ASSOCIATION"], "Societe": ["SOCIETE"]}
COPY_SUFFIXES = ["et Cie", "& Cie", "Services", "Associés", "et Fils", "& Fils"]
DECOY_WORDS = ["Groupe", "Holding", "Développement", "Participations", "Distribution", "International", "France"]
STREET_MAP = {"street": "Rue", "st": "Rue", "str": "Rue", "avenue": "Avenue", "ave": "Avenue", "av": "Avenue",
              "boulevard": "Boulevard", "blvd": "Boulevard", "drive": "Allée", "dr": "Allée",
              "road": "Route", "rd": "Route", "highway": "Route", "hwy": "Route", "lane": "Chemin", "ln": "Chemin",
              "court": "Impasse", "ct": "Impasse", "place": "Place", "pl": "Place", "plaza": "Place",
              "way": "Voie", "parkway": "Cours", "pkwy": "Cours", "trail": "Sentier", "square": "Square",
              "circle": "Square", "cir": "Square", "terrace": "Passage", "marg": "Rue", "nagar": "Quai"}
STREET_ABBREV = {"Rue": ["R.", "R"], "Avenue": ["Av.", "AV.", "Av", "Ave"], "Boulevard": ["Bd"],
                 "Impasse": ["Imp.", "Imp"], "Place": ["Pl.", "Pl"], "Chemin": ["Ch."], "Allée": ["All."],
                 "Route": ["Rte"], "Cours": ["Crs"], "Square": ["Sq"]}
ARTICLES = ["", "", "", "", "", "de ", "du ", "des ", "de la "]
REGIONS = ["Hauts-de-France", "Nouvelle-Aquitaine", "Pays de la Loire"]
DEPARTEMENTS = ["Nord", "Pas-de-Calais", "Gironde", "Loire-Atlantique"]
US_STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY "
                "NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR".split())
US_STATE_NAMES = {s.lower() for s in (
    "Alabama,Alaska,Arizona,Arkansas,California,Colorado,Connecticut,Delaware,Florida,Georgia,Hawaii,Idaho,Illinois,"
    "Indiana,Iowa,Kansas,Kentucky,Louisiana,Maine,Maryland,Massachusetts,Michigan,Minnesota,Mississippi,Missouri,"
    "Montana,Nebraska,Nevada,New Hampshire,New Jersey,New Mexico,New York,North Carolina,North Dakota,Ohio,Oklahoma,"
    "Oregon,Pennsylvania,Rhode Island,South Carolina,South Dakota,Tennessee,Texas,Utah,Vermont,Virginia,Washington,"
    "West Virginia,Wisconsin,Wyoming,District of Columbia,Puerto Rico").split(",")}
BRACKET_RE = re.compile(r"\((?:India|USA|US|U\.S\.A\.|America)\)", re.I)
IN_STATES = {s.lower() for s in (
    "Andhra Pradesh,Arunachal Pradesh,Assam,Bihar,Chhattisgarh,Goa,Gujarat,Haryana,Himachal Pradesh,Jharkhand,"
    "Karnataka,Kerala,Madhya Pradesh,Maharashtra,Manipur,Meghalaya,Mizoram,Nagaland,Odisha,Punjab,Rajasthan,"
    "Sikkim,Tamil Nadu,Telangana,Tripura,Uttar Pradesh,Uttarakhand,West Bengal,Delhi,Chandigarh,Puducherry,"
    "Jammu and Kashmir,Ladakh,Andaman and Nicobar Islands,Dadra and Nagar Haveli and Daman and Diu,Lakshadweep"
).split(",")}

# per-record style-operation rates. Sources: generator_params.json france_copy_ops_keypairs /
# france_label_free where available; otherwise the supplied-data format rates in vocab.json.
RATES = {
    "legal_drop": .158, "legal_add": .088, "legal_swap": .013, "legal_dotted": .13,
    "copy_suffix": .04, "street_abbrev": .40, "number_format": .20, "title_case_upper_addr": .60,
    "accent_drop": .30, "admin_departement": .31, "admin_drop": .34, "bracket_france_s1": .08,
    "bracket_keep_copy": .75, "s1_add_legal_if_none": .5,
}


def verified_vocab(vocab_counts: dict) -> tuple[dict, dict]:
    """Drop every token whose supplied-data count is zero; return kept vocab + audit."""
    def count(group, token):
        entry = vocab_counts.get(group, {}).get(token)
        if entry is None:
            entry = vocab_counts.get(group, {}).get(token + "|ci")
        return 0 if entry is None else int(sum(entry.values()))
    audit, dropped = {}, []
    forms = {}
    for form, weight in FR_FORMS.items():
        n = count("legal", form)
        audit[f"legal:{form}"] = n
        if n > 0:
            forms[form] = weight
        else:
            dropped.append(f"legal:{form}")
    dotted = {}
    for form, variants in DOTTED.items():
        keep = []
        for v in variants:
            n = count("legal", v)
            audit[f"legal_variant:{v}"] = n
            (keep if n > 0 else dropped).append(v if n > 0 else f"legal_variant:{v}")
        dotted[form] = keep
    def filt(group, tokens, tag):
        keep = []
        for t in tokens:
            n = count(group, t)
            audit[f"{tag}:{t}"] = n
            if n > 0:
                keep.append(t)
            else:
                dropped.append(f"{tag}:{t}")
        return keep
    suffixes = filt("suffix", COPY_SUFFIXES, "copy_suffix")
    decoys = filt("decoy", DECOY_WORDS, "decoy_word")
    types = sorted(set(STREET_MAP.values()))
    kept_types = set(filt("street", types, "street_type"))
    street_map = {k: v for k, v in STREET_MAP.items() if v in kept_types}
    abbrev = {}
    for full, variants in STREET_ABBREV.items():
        keep = [v for v in filt("street", variants, f"street_abbrev[{full}]")]
        if full in kept_types and keep:
            abbrev[full] = keep
    articles = [a for a in ARTICLES if not a] + [a + " " for a in filt("articles", [a.strip() for a in ARTICLES if a],
                                                                      "article")]
    regions = filt("admin", REGIONS, "region")
    deps = filt("admin", DEPARTEMENTS, "departement")
    audit["bracket:(France)"] = count("suffix", "(France)")
    vocab = {"forms": forms, "dotted": dotted, "suffixes": suffixes, "decoys": decoys, "street_map": street_map,
             "abbrev": abbrev, "articles": articles, "regions": regions, "departements": deps,
             "bracket": audit["bracket:(France)"] > 0}
    return vocab, {"counts_in_supplied_french_rows": audit, "dropped_zero_count": dropped}


# ---------------------------------------------------------------------------
# Text operations
# ---------------------------------------------------------------------------

class Frenchifier:
    def __init__(self, vocab: dict, rates: dict, seed: int):
        self.v, self.r, self.seed = vocab, rates, seed
        alts = "|".join(f"(?:{p})" for p, _ in LEGAL_MAP)
        self.legal_re = re.compile(rf"(?<![\w.])({alts})(?![\w])", re.I)
        self.legal_targets = [(re.compile(rf"^(?:{p})$", re.I), f) for p, f in LEGAL_MAP]
        fr_alts = sorted({f for f in vocab["forms"]} | {v for vs in vocab["dotted"].values() for v in vs},
                         key=len, reverse=True)
        self.fr_legal_re = re.compile(r"(?<![\w.])(" + "|".join(re.escape(x) for x in fr_alts) + r")(?![\w])")
        types = sorted(vocab["street_map"], key=len, reverse=True)
        self.street_re = re.compile(r"^(?P<lead>\s*)(?P<num>(?:\d+[A-Za-z]?(?:[-/]\d+)?\s+)?)(?P<body>.*?\S)\s+"
                                    r"(?P<type>" + "|".join(types) + r")\.?(?P<tail>(?:\s+(?:N|S|E|W|NE|NW|SE|SW))?)\s*$",
                                    re.I)
        full_types = sorted(vocab["abbrev"], key=len, reverse=True)
        self.fr_type_re = re.compile(r"(?<![\w.])(" + "|".join(re.escape(t) for t in full_types) + r")(?![\w])", re.I)
        self.num_re = re.compile(r"(?<![\w°#(])(\d{1,4})(?![\d])")
        self.forms = list(vocab["forms"])
        self.form_w = np.asarray([vocab["forms"][f] for f in self.forms], float)
        self.form_w /= self.form_w.sum()
        self.form_cdf = np.cumsum(self.form_w)

    # -- helpers --------------------------------------------------------------
    def pick_form(self, u: float) -> str:
        return self.forms[int(np.searchsorted(self.form_cdf, u * self.form_cdf[-1], side="right").clip(0, len(self.forms) - 1))]

    @staticmethod
    def core_key(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", fold(name).lower()).strip()

    def map_legal(self, name: str) -> tuple[str, str | None]:
        """Replace English/Indian legal phrases by French forms; return (core-without-legal, form)."""
        found = None
        name = BRACKET_RE.sub("(France)", name)
        def repl(m):
            nonlocal found
            for pat, form in self.legal_targets:
                if pat.match(m.group(1)) and form in self.v["forms"]:
                    found = found or form
                    return ""
            return m.group(0)
        core = self.legal_re.sub(repl, name)
        core = re.sub(r"\s{2,}", " ", re.sub(r"[\s,]+$", "", core)).strip()
        return (core if core else name), found

    def admin_token(self, comp: str) -> bool:
        c = comp.strip()
        return c.upper() in US_STATES or c.lower() in IN_STATES or c.lower() in US_STATE_NAMES

    def base_address(self, addr: str) -> tuple[list[str], int | None, str | None]:
        """Base map: street type -> French (before name, per-street article), state -> region.
        Returns components, index of the street component (or None), region (or None)."""
        comps = [c for c in (addr or "").split(",")]
        street_i, region = None, None
        for i, comp in enumerate(comps):
            m = self.street_re.match(comp)
            if m and street_i is None:
                ftype = self.v["street_map"][m.group("type").lower().rstrip(".")]
                body = m.group("body").strip()
                art = self.v["articles"][h32("art", body.lower()) % len(self.v["articles"])]
                upper = comp.upper() == comp and any(ch.isalpha() for ch in comp)
                text = f"{ftype} {art}{body}{m.group('tail')}"
                if upper:
                    text = text.upper()
                num = m.group("num").strip()
                comps[i] = f"{m.group('lead') or ' ' if i else ''}{num + ' ' if num else ''}{text}"
                street_i = i
        for i in range(len(comps) - 1, -1, -1):
            if self.admin_token(comps[i]):
                region = self.v["regions"][h32("region", comps[i].strip().lower()) % len(self.v["regions"])]
                lead = " " if i else ""
                comps[i] = lead + (region.upper() if comps[i].strip().isupper() and len(comps[i].strip()) > 2 else region)
                break
        return comps, street_i, region

    def entity(self, name: str, addr: str) -> dict:
        """S1-level (entity) decisions, keyed on text so records of the same core share them."""
        core, form = self.map_legal(name)
        key = self.core_key(core)
        added = False
        if form is None and (h32("addlegal", key) % 1000) / 1000 < self.r["s1_add_legal_if_none"]:
            form = self.pick_form((h32("form", key) % 100000) / 100000)
            added = True
        bracket = self.v["bracket"] and (h32("bracket", key) % 1000) / 1000 < self.r["bracket_france_s1"]
        return {"core": core, "form": form, "added": added, "bracket": bracket, "key": key}

    @staticmethod
    def insert_bracket(core: str) -> str:
        words = core.split()
        if len(words) < 2:
            return core + " (France)"
        return " ".join(words[:1] + ["(France)"] + words[1:])

    def compose_name(self, core: str, form: str | None, bracket: bool, upper: bool) -> str:
        name = self.insert_bracket(core) if bracket else core
        if form:
            name = f"{name} {form}"
        return name.upper() if upper else name

    # -- style ops (every record, same rates, label-blind) ----------------------
    def style_record(self, core: str, form: str | None, bracket: bool, comps: list[str], street_i, upper_name: bool,
                     rng: random.Random, ops: Counter, is_s1: bool = False, legal_fixed: bool = False) -> tuple[str, str]:
        r = self.r
        if legal_fixed:
            ops["legal_fixed_same_number_positive"] += 1
            if form and self.v["dotted"].get(form) and rng.random() < r["legal_dotted"]:
                form = rng.choice(self.v["dotted"][form]); ops["legal_dotted_or_case"] += 1
        elif not is_s1:
            if form and rng.random() < r["legal_drop"]:
                form = None; ops["legal_dropped"] += 1
            elif form and rng.random() < r["legal_swap"]:
                form = rng.choice([f for f in self.forms if f != form]); ops["legal_swapped"] += 1
            elif not form and rng.random() < r["legal_add"]:
                form = self.pick_form(rng.random()); ops["legal_added"] += 1
            if form and self.v["dotted"].get(form) and rng.random() < r["legal_dotted"]:
                form = rng.choice(self.v["dotted"][form]); ops["legal_dotted_or_case"] += 1
            if bracket and rng.random() > r["bracket_keep_copy"]:
                bracket = False; ops["bracket_dropped"] += 1
        name = self.compose_name(core, form, bracket, upper_name)
        if not is_s1 and self.v["suffixes"] and rng.random() < r["copy_suffix"]:
            suf = rng.choice(self.v["suffixes"])
            name = f"{name} {suf.upper() if upper_name else suf}"; ops["copy_suffix"] += 1
        comps = list(comps)
        if not is_s1:
            if street_i is not None:
                comp = comps[street_i]
                if rng.random() < r["street_abbrev"]:
                    m = self.fr_type_re.search(comp)
                    if m:
                        full = next(k for k in self.v["abbrev"] if fold(k).lower() == fold(m.group(1)).lower())
                        ab = rng.choice(self.v["abbrev"][full])
                        comp = comp[:m.start()] + (ab.upper() if m.group(1).isupper() else ab) + comp[m.end():]
                        ops["street_abbrev"] += 1
                if rng.random() < r["number_format"]:
                    m = self.num_re.search(comp)
                    if m:
                        n = m.group(1)
                        style = rng.choice(["N°", "No", "#", "paren", "zero", "bis", "ter", "dash"])
                        new = {"N°": f"N°{n}", "No": f"No {n}", "#": f"#{n}", "paren": f"({n})",
                               "zero": n.zfill(len(n) + rng.choice([1, 2])), "bis": f"{n} bis", "ter": f"{n} ter",
                               "dash": f"{n} -"}[style]
                        comp = comp[:m.start()] + new + comp[m.end():]
                        ops[f"number_format_{style}"] += 1
                comps[street_i] = comp
            # admin: region -> département or dropped
            for i in range(len(comps) - 1, -1, -1):
                tok = comps[i].strip()
                if tok and fold(tok).lower() in {fold(x).lower() for x in self.v["regions"]}:
                    u = rng.random()
                    if u < r["admin_departement"] and self.v["departements"]:
                        dep = rng.choice(self.v["departements"])
                        comps[i] = (" " if i else "") + (dep.upper() if tok.isupper() else dep); ops["admin_departement"] += 1
                    elif u < r["admin_departement"] + r["admin_drop"] and len(comps) > 1:
                        comps[i] = None; ops["admin_dropped"] += 1
                    break
        comps = [c for c in comps if c is not None]
        addr = ",".join(comps).strip()
        if not is_s1:
            if addr.isupper() and rng.random() < r["title_case_upper_addr"]:
                addr = re.sub(r"[A-Za-zÀ-ÿ]+", lambda m: m.group(0).capitalize(), addr.lower()); ops["upper_to_title_addr"] += 1
            if rng.random() < r["accent_drop"]:
                if fold(name) != name or fold(addr) != addr:
                    ops["accent_dropped"] += 1
                name, addr = fold(name), fold(addr)
        return name, addr

    def shift_number(self, addr: str, street_i, delta: int, comps: list[str]) -> list[str] | None:
        comps = list(comps)
        idx = [street_i] if street_i is not None else list(range(len(comps)))
        for i in idx:
            m = self.num_re.search(comps[i])
            if m:
                comps[i] = comps[i][:m.start()] + str(int(m.group(1)) + delta) + comps[i][m.end():]
                return comps
        return None


# ---------------------------------------------------------------------------
# World construction
# ---------------------------------------------------------------------------

def make_world(tag: str, s1_ids: np.ndarray, rows: np.ndarray, left: np.ndarray, right: np.ndarray,
               label: np.ndarray, tree: np.ndarray, blend: np.ndarray, ref: pd.DataFrame, pool_text,
               rec_owner: np.ndarray, true_count: np.ndarray, fz: Frenchifier, seed: int, args) -> dict:
    """Frenchify one world. left/right are arena indices of the kept pairs of s1_ids."""
    started = time.monotonic()
    ops_pos, ops_neg, ops_s1 = Counter(), Counter(), Counter()
    s1_set = np.zeros(len(ref), bool); s1_set[s1_ids] = True
    recs = np.unique(right)
    owners = rec_owner[recs]
    ent_cache: dict[int, dict] = {}
    names, addrs = ref.business_name.to_numpy(), ref.business_address.to_numpy()

    def ent(i: int) -> dict:
        e = ent_cache.get(i)
        if e is None:
            n = names[i] or ""
            e = fz.entity(n, addrs[i] or "")
            e["upper"] = n.isupper()
            ent_cache[i] = e
        return e

    # S1 text
    s1_name, s1_addr, s1_street, s1_comps = {}, {}, {}, {}
    for i in s1_ids:
        e = ent(int(i))
        comps, street_i, _ = fz.base_address(addrs[i] or "")
        rng = random.Random(h32(seed, tag, "s1", i))
        nm, ad = fz.style_record(e["core"], e["form"], e["bracket"], comps, street_i, e["upper"], rng, ops_s1, is_s1=True)
        s1_name[int(i)], s1_addr[int(i)], s1_street[int(i)], s1_comps[int(i)] = nm, ad, street_i, comps
        if e["added"]:
            ops_s1["s1_legal_added_by_hash"] += 1
        if e["bracket"]:
            ops_s1["s1_bracket_france"] += 1
    log(f"[{tag}] S1 text {len(s1_ids)}")

    # record text
    rec_ids = pool_text["entity_id"][recs]
    rec_names = pool_text["business_name"][recs]
    rec_addrs = pool_text["business_address"][recs]
    out_name, out_addr = [], []
    rec_kind_pos = np.zeros(len(recs), bool)
    owned_in_world = (owners >= 0) & s1_set[np.maximum(owners, 0)]
    rec_kind_pos[owned_in_world] = True
    for j in range(len(recs)):
        rn, ra = rec_names[j] or "", rec_addrs[j] or ""
        rng = random.Random(h32(seed, tag, "rec", rec_ids[j]))
        core, form = fz.map_legal(rn)
        o = int(owners[j])
        if o >= 0:
            e = ent(o)
            if form is None and e["added"]:
                form = e["form"]
            elif form is not None and e["form"] is not None and not e["added"]:
                form = e["form"] if form == e["form"] else form
            bracket = e["bracket"]
        else:
            key = fz.core_key(core)
            if form is None and (h32("addlegal", key) % 1000) / 1000 < fz.r["s1_add_legal_if_none"]:
                form = fz.pick_form((h32("form", key) % 100000) / 100000)
            bracket = fz.v["bracket"] and (h32("bracket", key) % 1000) / 1000 < fz.r["bracket_france_s1"]
        comps, street_i, _ = fz.base_address(ra)
        counter = ops_pos if rec_kind_pos[j] else ops_neg
        nm, ad = fz.style_record(core, form, bracket, comps, street_i, rn.isupper(), rng, counter)
        out_name.append(nm); out_addr.append(ad)
    log(f"[{tag}] record text {len(recs)} ({time.monotonic() - started:.0f}s)")

    # local indexing: world S1s first, then siblings; world records then synthetic records
    s1_local = {int(i): k for k, i in enumerate(s1_ids)}
    rec_local = np.full(len(rec_owner), -1, np.int64)
    rec_local[recs] = np.arange(len(recs))
    ref_rows = [{"entity_id": f"{tag}-{ref.entity_id.iat[i]}", "business_name": s1_name[int(i)],
                 "business_address": s1_addr[int(i)], "country": "France", "role": f"{tag}_s1",
                 "source_idx": int(i), "origin_country": ref.country.iat[i]} for i in s1_ids]
    pool_rows = [{"entity_id": f"{tag}-{rec_ids[j]}", "business_name": out_name[j], "business_address": out_addr[j],
                  "country": "France", "source_idx": int(recs[j]), "origin_country": pool_text["country"][recs[j]]}
                 for j in range(len(recs))]
    # owners: in-world owner -> local S1; others -> stub rows appended after siblings
    p_left = np.asarray([s1_local[int(x)] for x in left], np.int64)
    p_right = rec_local[right]
    p_label = label.astype(bool)
    p_tree, p_blend = tree.astype(np.float32), blend.astype(np.float32)
    p_kind = np.where(p_label, K_REAL_POS, K_REAL_NEG).astype(np.int8)
    p_row = rows.astype(np.int64)

    # synthetic
    syn_left, syn_right, syn_label, syn_kind = [], [], [], []
    sib_rows, sib_true = [], []
    # local owner: world S1 index; -2 = owned outside the world (stub later); -1 = unowned
    rec_owner_local = [s1_local[int(o)] if w else (-2 if o >= 0 else -1) for o, w in zip(owners, owned_in_world)]
    s1_true_records = {}
    for jj, o in enumerate(owners):
        if owned_in_world[jj]:
            s1_true_records.setdefault(int(o), []).append(jj)
    street_bodies = []
    for i in s1_ids[: min(len(s1_ids), 50_000)]:
        c = s1_comps[int(i)]; si = s1_street[int(i)]
        if si is not None:
            street_bodies.append(re.sub(r"^\s*(?:\d+\S*\s+)?", "", c[si]))
    extra_true = np.zeros(len(s1_ids), np.int64)
    for i in s1_ids:
        i = int(i)
        e = ent(i)
        rng = random.Random(h32(seed, tag, "syn", i))
        comps, si = s1_comps[i], s1_street[i]
        k = s1_local[i]
        # (a) legal-form swap + delta
        if rng.random() < args.p_legal_swap:
            shifted = fz.shift_number("", si, draw_delta(rng), comps)
            if shifted is not None:
                other = [f for f in fz.forms if f != e["form"]]
                nm, ad = fz.style_record(e["core"], rng.choice(other), e["bracket"], shifted, si, e["upper"], rng, ops_neg)
                pool_rows.append({"entity_id": f"{tag}-SYNL-{ref.entity_id.iat[i]}", "business_name": nm,
                                  "business_address": ad, "country": "France", "source_idx": -1,
                                  "origin_country": ref.country.iat[i]})
                rec_owner_local.append(-1)
                syn_left.append(k); syn_right.append(len(pool_rows) - 1); syn_label.append(False); syn_kind.append(K_LEGAL_SWAP)
        # (b) decoy word + number shift
        if fz.v["decoys"] and rng.random() < args.p_decoy_word:
            shifted = fz.shift_number("", si, draw_delta(rng), comps)
            if shifted is not None:
                word = rng.choice(fz.v["decoys"])
                core = f"{e['core']} {word.upper() if e['upper'] else word}"
                nm, ad = fz.style_record(core, e["form"], e["bracket"], shifted, si, e["upper"], rng, ops_neg)
                pool_rows.append({"entity_id": f"{tag}-SYND-{ref.entity_id.iat[i]}", "business_name": nm,
                                  "business_address": ad, "country": "France", "source_idx": -1,
                                  "origin_country": ref.country.iat[i]})
                rec_owner_local.append(-1)
                syn_left.append(k); syn_right.append(len(pool_rows) - 1); syn_label.append(False); syn_kind.append(K_DECOY_WORD)
        # (a+) extra positive copy: same number/address, legal form dropped (S1 has one) or added (S1 has none)
        if rng.random() < args.p_legal_positive:
            pform = None if e["form"] else fz.pick_form(rng.random())
            nm, ad = fz.style_record(e["core"], pform, e["bracket"], comps, si, e["upper"], rng, ops_pos, is_s1=False,
                                     legal_fixed=True)
            pool_rows.append({"entity_id": f"{tag}-SYNP-{ref.entity_id.iat[i]}", "business_name": nm,
                              "business_address": ad, "country": "France", "source_idx": -1,
                              "origin_country": ref.country.iat[i]})
            rec_owner_local.append(k)
            syn_left.append(k); syn_right.append(len(pool_rows) - 1); syn_label.append(True); syn_kind.append(K_LEGAL_POS)
            extra_true[k] += 1
        # (c) sibling S1: same core + number + city, other street, other legal form
        if si is not None and street_bodies and rng.random() < args.p_sibling:
            m = fz.num_re.search(comps[si])
            body = rng.choice(street_bodies)
            if m and fold(body).lower() not in fold(comps[si]).lower():
                sib_comps = list(comps)
                sib_comps[si] = (" " if si else "") + m.group(1) + " " + body.strip()
                other = [f for f in fz.forms if f != e["form"]]
                sform = rng.choice(other)
                sname, saddr = fz.style_record(e["core"], sform, e["bracket"], sib_comps, si, e["upper"], rng,
                                               Counter(), is_s1=True)
                sib_local = len(s1_ids) + len(sib_rows)
                n_copies = rng.choice([1, 1, 2, 2, 3, 4])
                sib_rows.append({"entity_id": f"{tag}-SIB-{ref.entity_id.iat[i]}", "business_name": sname,
                                 "business_address": saddr, "country": "France", "role": f"{tag}_sibling",
                                 "source_idx": i, "origin_country": ref.country.iat[i]})
                sib_true.append(n_copies)
                copies = []
                for c in range(n_copies):
                    upper = rng.random() < .3
                    nm, ad = fz.style_record(e["core"], sform, e["bracket"], sib_comps, si, upper or e["upper"], rng, ops_pos)
                    if upper:
                        ad = ad.upper()
                    pool_rows.append({"entity_id": f"{tag}-SIBC{c}-{ref.entity_id.iat[i]}", "business_name": nm,
                                      "business_address": ad, "country": "France", "source_idx": -1,
                                      "origin_country": ref.country.iat[i]})
                    rec_owner_local.append(sib_local)
                    copies.append(len(pool_rows) - 1)
                for r_ in copies:
                    syn_left.append(sib_local); syn_right.append(r_); syn_label.append(True); syn_kind.append(K_SIB_POS)
                    syn_left.append(k); syn_right.append(r_); syn_label.append(False); syn_kind.append(K_ORIG_VS_SIB)
                for jj in s1_true_records.get(i, []):
                    syn_left.append(sib_local); syn_right.append(jj); syn_label.append(False); syn_kind.append(K_SIB_VS_ORIG)
    log(f"[{tag}] synthetic pairs {len(syn_left)}; siblings {len(sib_rows)}")

    # stubs for owners outside the world
    stub_src = sorted({int(o) for o, w in zip(owners, owned_in_world) if o >= 0 and not w})
    stub_base = len(s1_ids) + len(sib_rows)
    stub_local = {o: stub_base + t for t, o in enumerate(stub_src)}
    owner_arr = np.asarray(rec_owner_local, np.int64)
    for jj in np.flatnonzero(owner_arr[:len(recs)] == -2):
        owner_arr[jj] = stub_local[int(owners[jj])]
    stub_rows = [{"entity_id": f"{tag}-STUB-{ref.entity_id.iat[o]}", "business_name": names[o], "business_address": addrs[o],
                  "country": ref.country.iat[o], "role": f"{tag}_owner_stub", "source_idx": o,
                  "origin_country": ref.country.iat[o]} for o in stub_src]
    reference = pd.DataFrame(ref_rows + sib_rows + stub_rows)
    pool = pd.DataFrame(pool_rows)
    tc = np.concatenate([true_count[s1_ids] + extra_true, np.asarray(sib_true, np.int64), true_count[stub_src] if stub_src else []]).astype(np.int16)
    n_syn = len(syn_left)
    left_all = np.concatenate([p_left, np.asarray(syn_left, np.int64)])
    right_all = np.concatenate([p_right, np.asarray(syn_right, np.int64)])
    label_all = np.concatenate([p_label, np.asarray(syn_label, bool)])
    kind_all = np.concatenate([p_kind, np.asarray(syn_kind, np.int8)])
    tree_all = np.concatenate([p_tree, np.full(n_syn, SYN_SCORE, np.float32)])
    blend_all = np.concatenate([p_blend, np.full(n_syn, np.float32(args.syn_blend), np.float32)])
    row_all = np.concatenate([p_row, np.full(n_syn, -1, np.int64)])
    if not np.array_equal(label_all, owner_arr[right_all] == left_all):
        bad = np.flatnonzero(label_all != (owner_arr[right_all] == left_all))[:5]
        raise ValueError(f"[{tag}] owners do not reproduce labels, e.g. rows {bad}")
    ops_pos_pairs = Counter()
    log(f"[{tag}] done: S1 {len(s1_ids)} siblings {len(sib_rows)} stubs {len(stub_rows)} records {len(pool)} "
        f"pairs {len(left_all)} ({time.monotonic() - started:.0f}s)")
    return {"reference": reference, "pool": pool, "left": left_all, "right": right_all, "label": label_all,
            "kind": kind_all, "tree": tree_all, "blend": blend_all, "source_row": row_all, "owner": owner_arr,
            "true_count": tc, "n_s1": len(s1_ids), "n_sib": len(sib_rows), "n_stub": len(stub_rows),
            "ops": {"s1": dict(ops_s1), "records_owned_by_world_s1_incl_sibling_copies": dict(ops_pos),
                    "records_other": dict(ops_neg)}}


def _take_text(path: Path, positions: np.ndarray) -> pd.DataFrame:
    table = pq.read_table(path, columns=TEXT_COLUMNS)
    return table.take(pa.array(positions, type=pa.int64())).to_pandas()


def build(args) -> dict:
    started = time.monotonic()
    out = Path(args.output)
    arena_out, label_out = out / "arena", out / "labels"
    for path in (arena_out, label_out):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"output must be new or empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
    arena, labels_dir = Path(args.arena), Path(args.labels)
    summary = json.loads((labels_dir / "labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise ValueError("refusing arena labels that read HOLDOUT-2")
    forbidden, sealed_sha = sealed_ids(Path(args.sealed_reservation))
    vocab_counts = json.loads(Path(args.vocab_counts).read_text())
    vocab, vocab_audit = verified_vocab(vocab_counts)
    log(f"vocab kept: {json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in vocab.items()}, ensure_ascii=False)}")
    log(f"vocab dropped (zero count): {vocab_audit['dropped_zero_count']}")
    fz = Frenchifier(vocab, RATES, args.seed)

    part = np.load(args.arena_partition)
    n_ref = len(part)
    rng = np.random.default_rng(args.seed)
    a_ids = np.flatnonzero(part == CODE_A)
    perm = rng.permutation(a_ids)
    per_s1 = summary["positive_pairs"] / n_ref
    fr_pos_est = args.french_s1 * per_s1 * (1 + args.p_sibling * 0.6)
    n_rp = int(np.ceil(fr_pos_est * (1 - args.french_share) / args.french_share / per_s1))
    fr_ids = np.sort(perm[:args.french_s1])
    tu_ids = np.sort(perm[args.french_s1:args.french_s1 + args.tune_s1])
    rp_ids = np.sort(perm[args.french_s1 + args.tune_s1:args.french_s1 + args.tune_s1 + n_rp])
    if (Path(args.monitor_from) / "arena/reference.parquet").exists():
        mix_ref = pq.read_table(Path(args.monitor_from) / "arena/reference.parquet", columns=["role", "source_idx", "source"]).to_pandas()
        mon_ids = np.sort(mix_ref.source_idx.to_numpy()[(mix_ref.role == "monitor_b_rest").to_numpy()])
    else:
        # reproduce ber.france_mix_arena's 26 Sep draw (seed 20260926; 254,312 replay S1s drawn first)
        mrng = np.random.default_rng(20260926)
        mrng.choice(np.flatnonzero(part == CODE_A), 254_312, replace=False)
        mon_ids = np.sort(mrng.choice(np.flatnonzero(part == CODE_B_REST), 20_000, replace=False))
        log("monitor S1s reproduced from the france_mix_arena seed (no france_mix reference available)")
    if len(mon_ids) != 20_000 or np.any(part[mon_ids] != CODE_B_REST):
        raise ValueError("monitor S1s are not the 20,000 B_REST S1s of the 26 Sep france_mix arena")
    log(f"FR {len(fr_ids)}  TU {len(tu_ids)}  replay {len(rp_ids)}  monitor {len(mon_ids)} (per-S1 positives {per_s1:.3f})")

    role = np.zeros(n_ref, np.int8)
    role[fr_ids], role[tu_ids], role[rp_ids], role[mon_ids] = 1, 2, 3, 4
    pairs_file = pq.ParquetFile(arena / "pairs.parquet")
    sel = {1: [], 2: [], 3: [], 4: []}
    offset = 0
    for batch in pairs_file.iter_batches(batch_size=CHUNK, columns=["left_idx", "right_idx"]):
        lft = batch.column(0).to_numpy()
        rl = role[lft]
        for code in sel:
            sel[code].append(np.flatnonzero(rl == code) + offset)
        offset += len(lft)
    sel = {c: np.concatenate(v) for c, v in sel.items()}
    log(f"pair scan done: {offset} rows; " + ", ".join(f"{c}:{len(v)}" for c, v in sel.items()))
    left_src = pq.read_table(arena / "pairs.parquet", columns=["left_idx"]).column(0).to_numpy()
    right_src = pq.read_table(arena / "pairs.parquet", columns=["right_idx"]).column(0).to_numpy()
    label_src = np.load(labels_dir / "label.npy", mmap_mode="r")
    tree_src = np.load(arena / "base_probabilities.npy", mmap_mode="r")
    blend_src = np.load(arena / "probabilities.npy", mmap_mode="r")
    owner_src = np.load(labels_dir / "rec_owner.npy")
    true_src = np.load(labels_dir / "true_count.npy")
    if offset != len(tree_src) or offset != len(label_src):
        raise ValueError("arena pairs and scores/labels are not aligned")

    ref = pq.read_table(arena / "reference.parquet", columns=TEXT_COLUMNS).to_pandas()
    # FR/TU keep positives + tree-gated negatives (only these are ever sampled / scored)
    worlds = {}
    need_rec = []
    for code in (1, 2):
        rows = sel[code]
        lab = np.asarray(label_src[rows], bool)
        tr = np.asarray(tree_src[rows], np.float32)
        keep = lab | (tr >= args.tree_gate)
        if code == 1 and args.fr_untyped_negative_keep < 1:
            # thin the FR real decoys that fast_trainer would only use as random (untyped) negatives,
            # so French negatives:positives stays close to the replay ratio; TU keeps everything
            bl = np.asarray(blend_src[rows], np.float32)
            untyped = ~lab & (bl < .3)
            drop = untyped & (np.random.default_rng(args.seed + 1).random(len(rows)) >= args.fr_untyped_negative_keep)
            log(f"FR untyped real negatives {int((untyped & keep).sum())}; dropped {int((drop & keep).sum())}")
            keep &= ~drop
        worlds[code] = rows[keep]
        need_rec.append(right_src[worlds[code]])
    need_rec = np.unique(np.concatenate(need_rec)).astype(np.int64)
    ptab = pq.read_table(arena / "pool.parquet", columns=TEXT_COLUMNS).take(pa.array(need_rec, type=pa.int64()))
    pool_text = {c: np.empty(len(owner_src), object) for c in TEXT_COLUMNS}
    for c in TEXT_COLUMNS:
        pool_text[c][need_rec] = np.asarray(ptab.column(c).to_pylist(), object)
    del ptab
    log(f"world records to frenchify: {len(need_rec)}")
    built = {}
    for code, tag, seed in ((1, "FR", args.seed), (2, "TU", args.seed + 7919)):
        rows = worlds[code]
        built[tag] = make_world(tag, fr_ids if code == 1 else tu_ids, rows, left_src[rows].astype(np.int64),
                                right_src[rows].astype(np.int64), np.asarray(label_src[rows], bool),
                                np.asarray(tree_src[rows], np.float32), np.asarray(blend_src[rows], np.float32),
                                ref, pool_text, owner_src, true_src, fz, seed, args)

    # --- replay + monitor (original text) --------------------------------------
    rows_a = np.sort(np.concatenate([sel[3], sel[4]]))
    left_a, right_a = left_src[rows_a].astype(np.int64), right_src[rows_a].astype(np.int64)
    label_a = np.asarray(label_src[rows_a], bool)
    tree_a = np.asarray(tree_src[rows_a], np.float32)
    blend_a = np.asarray(blend_src[rows_a], np.float32)
    ar_rec = np.unique(right_a)
    ar_owner = owner_src[ar_rec].astype(np.int64)
    chosen = np.zeros(n_ref, bool); chosen[rp_ids] = True; chosen[mon_ids] = True
    stubs = np.setdiff1d(np.unique(ar_owner[ar_owner >= 0]), np.flatnonzero(chosen))
    ar_s1 = np.concatenate([rp_ids, mon_ids, stubs])

    # --- assemble ------------------------------------------------------------------
    fr, tu = built["FR"], built["TU"]
    ref_parts, pool_parts = [], []
    s1_off, rec_off = 0, 0
    L, R, LAB, TREE, BLEND, KIND, SRCROW, WORLD = [], [], [], [], [], [], [], []
    owners, tcs, parts, roles = [], [], [], []
    for tag, w, pcode in (("FR", fr, CODE_A), ("TU", tu, CODE_SLICE)):
        n_main = w["n_s1"] + w["n_sib"]
        ref_parts.append(w["reference"].assign(source=np.uint8(2 if tag == "FR" else 3)))
        pool_parts.append(w["pool"].assign(source=np.uint8(2 if tag == "FR" else 3)))
        L.append(w["left"] + s1_off); R.append(w["right"] + rec_off); LAB.append(w["label"])
        TREE.append(w["tree"]); BLEND.append(w["blend"]); KIND.append(w["kind"]); SRCROW.append(w["source_row"])
        WORLD.append(np.full(len(w["left"]), 1 if tag == "FR" else 2, np.int8))
        owners.append(np.where(w["owner"] >= 0, w["owner"] + s1_off, -1))
        tcs.append(w["true_count"])
        parts.append(np.r_[np.full(n_main, pcode), np.full(w["n_stub"], CODE_A)].astype(np.uint8))
        roles.append(w["reference"].role.to_numpy())
        s1_off += len(w["reference"]); rec_off += len(w["pool"])
    new_a = np.full(n_ref, -1, np.int64); new_a[ar_s1] = s1_off + np.arange(len(ar_s1))
    rec_a = np.full(len(owner_src), -1, np.int64); rec_a[ar_rec] = rec_off + np.arange(len(ar_rec))
    ref_a = _take_text(arena / "reference.parquet", ar_s1)
    ref_a["role"] = np.r_[np.full(len(rp_ids), "replay_a"), np.full(len(mon_ids), "monitor_b_rest"),
                          np.full(len(stubs), "owner_stub")]
    ref_a["source_idx"] = ar_s1
    ref_a["origin_country"] = ref_a.country
    ref_a["source"] = np.uint8(1)
    pool_a = _take_text(arena / "pool.parquet", ar_rec)
    pool_a["source_idx"] = ar_rec
    pool_a["origin_country"] = pool_a.country
    pool_a["source"] = np.uint8(1)
    ref_parts.append(ref_a); pool_parts.append(pool_a)
    L.append(new_a[left_a]); R.append(rec_a[right_a]); LAB.append(label_a); TREE.append(tree_a); BLEND.append(blend_a)
    KIND.append(np.where(label_a, K_REAL_POS, K_REAL_NEG).astype(np.int8)); SRCROW.append(rows_a.astype(np.int64))
    WORLD.append(np.zeros(len(rows_a), np.int8))
    owners.append(np.where(ar_owner >= 0, new_a[np.maximum(ar_owner, 0)], -1))
    tcs.append(true_src[ar_s1].astype(np.int16))
    parts.append(np.r_[np.full(len(rp_ids), CODE_A), np.full(len(mon_ids), CODE_B_REST), np.full(len(stubs), CODE_A)].astype(np.uint8))
    roles.append(ref_a.role.to_numpy())

    left = np.concatenate(L); right = np.concatenate(R); label = np.concatenate(LAB)
    tree = np.concatenate(TREE).astype(np.float32); blend = np.concatenate(BLEND).astype(np.float32)
    kind = np.concatenate(KIND); srcrow = np.concatenate(SRCROW); world = np.concatenate(WORLD)
    owner_new = np.concatenate(owners).astype(np.int32)
    true_count = np.concatenate(tcs).astype(np.int16)
    partition = np.concatenate(parts).astype(np.uint8)
    reference = pd.concat(ref_parts, ignore_index=True)
    pool = pd.concat(pool_parts, ignore_index=True)
    if (left < 0).any() or (right < 0).any() or len(owner_new) != len(pool) or len(true_count) != len(reference):
        raise ValueError("unmapped index or misaligned arrays")
    order = np.lexsort((right, left))
    left, right, label, tree, blend = left[order], right[order], label[order], tree[order], blend[order]
    kind, srcrow, world = kind[order], srcrow[order], world[order]
    if len(left) > 1 and np.any((left[1:] == left[:-1]) & (right[1:] <= right[:-1])):
        raise ValueError("duplicate (left, right) pair")
    if not np.array_equal(label, owner_new[right] == left):
        raise ValueError("owners do not reproduce the pair labels")
    if np.any(np.bincount(left[label], minlength=len(true_count)) > true_count):
        raise ValueError("retrieved positives exceed true_count")
    dup_ref = int(reference.entity_id.duplicated().sum())
    dup_pool = int(pool.entity_id.duplicated().sum())
    if dup_ref or dup_pool:
        raise ValueError(f"duplicate IDs: reference {dup_ref}, pool {dup_pool}")
    if reference.entity_id.isin(forbidden).any():
        raise ValueError("sealed reference IDs present")

    for col in ("entity_id", "business_name", "business_address", "country", "role", "origin_country"):
        reference[col] = reference[col].astype(str)
    for col in ("entity_id", "business_name", "business_address", "country", "origin_country"):
        pool[col] = pool[col].fillna("").astype(str)
    reference[TEXT_COLUMNS + ["role", "source_idx", "origin_country", "source"]].to_parquet(
        arena_out / "reference.parquet", index=False, row_group_size=1_000_000)
    pool[TEXT_COLUMNS + ["source_idx", "origin_country", "source"]].to_parquet(
        arena_out / "pool.parquet", index=False, row_group_size=1_000_000)
    pq.write_table(pa.table({"left_idx": left.astype(np.int32), "right_idx": right.astype(np.int32),
                             "world": world, "kind": kind, "source_pair_row": srcrow}),
                   arena_out / "pairs.parquet", row_group_size=1_000_000)
    _npy(arena_out / "v1_blend.npy", blend)
    _npy(arena_out / "v1_tree.npy", tree)
    names = ("reference.parquet", "pool.parquet", "pairs.parquet", "v1_blend.npy", "v1_tree.npy")
    manifest = {"schema": 1, "source": "ber.frenchify_arena: train-derived French-operation arena (option A) + A replay + "
                "B_REST monitor + Frenchified tuning world (SLICE)",
                "references": int(len(reference)), "pool_records": int(len(pool)), "pairs": int(len(left)),
                "labels_read": False, "holdout2_excluded": True, "pair_order": "left_then_right",
                "pair_selection": ("FR/TU: real positives + real negatives with arena v1 tree >= gate + synthetic "
                                   "negatives/siblings; replay/monitor: all candidate pairs; no truth injected into "
                                   "real candidate lists (synthetic records exist only as their own pairs)"),
                "sha256": {name: file_sha256(arena_out / name) for name in names}}
    _json(arena_out / "manifest.json", manifest)
    _npy(label_out / "label.npy", label)
    _npy(label_out / "true_count.npy", true_count)
    _npy(label_out / "rec_owner.npy", owner_new)
    _json(label_out / "labels_summary.json", {
        "pairs": int(len(label)), "positive_pairs": int(label.sum()), "true_links": int(true_count.sum()),
        "singletons": int((true_count == 0).sum()), "holdout2_rows_read": 0,
        "labels": "arena TRAIN labels remapped + synthetic labels by construction"})
    _npy(out / "s1_partition.npy", partition)
    _npy(out / "eval_s1.npy", partition != CODE_A)
    _npy(out / "b_rest_s1.npy", partition == CODE_B_REST)
    _npy(out / "tune_s1.npy", partition == CODE_SLICE)

    train = partition[left] == CODE_A
    fr_rows = world == 1
    kinds = {KIND_NAMES[k]: {"train_FR": int(((kind == k) & fr_rows & train).sum()),
                             "tune_TU": int(((kind == k) & (world == 2)).sum())} for k in range(len(KIND_NAMES))}
    counts = {
        "references": int(len(reference)), "pool_records": int(len(pool)), "pairs": int(len(left)),
        "fr_s1": fr["n_s1"], "fr_siblings": fr["n_sib"], "fr_owner_stubs": fr["n_stub"],
        "tu_s1": tu["n_s1"], "tu_siblings": tu["n_sib"], "tu_owner_stubs": tu["n_stub"],
        "replay_a_s1": int(len(rp_ids)), "monitor_b_rest_s1": int(len(mon_ids)), "owner_stub_s1": int(len(stubs)),
        "partition": {n: int((partition == c).sum()) for c, n in ((CODE_A, "A"), (CODE_B_REST, "B_REST"), (CODE_SLICE, "SLICE_tune"))},
        "train_A": {"pairs": int(train.sum()), "positives": int((label & train).sum()),
                    "french_positives": int((label & train & fr_rows).sum()),
                    "replay_positives": int((label & train & (world == 0)).sum()),
                    "french_negatives": int((~label & train & fr_rows).sum()),
                    "replay_negatives": int((~label & train & (world == 0)).sum())},
        "tune": {"pairs": int((world == 2).sum()), "positives": int((label & (world == 2)).sum()),
                 "true_links": int(true_count[partition == CODE_SLICE].sum())},
        "b_rest_monitor": {"pairs": int((partition[left] == CODE_B_REST).sum()),
                           "positives": int((label & (partition[left] == CODE_B_REST)).sum())},
        "pair_kinds": kinds,
    }
    tp = counts["train_A"]
    tp["french_positive_share"] = tp["french_positives"] / max(1, tp["positives"])
    if tp["french_positive_share"] > .5:
        raise ValueError(f"French share of training positives {tp['french_positive_share']:.3f} > 0.5")
    # examples (train FR only) for the report
    ex = []
    exrng = np.random.default_rng(1)
    for k in range(len(KIND_NAMES)):
        cand = np.flatnonzero((kind == k) & fr_rows)
        for p in exrng.choice(cand, min(3, len(cand)), replace=False) if len(cand) else []:
            ex.append({"kind": KIND_NAMES[k], "label": bool(label[p]),
                       "s1": f"{reference.business_name.iat[left[p]]} | {reference.business_address.iat[left[p]]}",
                       "record": f"{pool.business_name.iat[right[p]]} | {pool.business_address.iat[right[p]]}"})
    report = {
        "state": "complete", "seed": args.seed, "tune_seed": args.seed + 7919, "elapsed_s": round(time.monotonic() - started, 1),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"},
        "rates": RATES, "deltas": DELTAS, "even_deltas": EVEN_DELTAS, "synthetic_tree": float(SYN_SCORE), "synthetic_blend": args.syn_blend,
        "vocab_kept": vocab, "vocab_audit": vocab_audit,
        "inputs": {"arena_partition_sha256": file_sha256(Path(args.arena_partition)),
                   "sealed_reservation_sha256": sealed_sha, "labels_summary_sha256": file_sha256(labels_dir / "labels_summary.json"),
                   "vocab_counts_sha256": file_sha256(Path(args.vocab_counts))},
        "counts": counts,
        "ops": {"FR": fr["ops"], "TU": tu["ops"]},
        "split_ids_sha256": {name: file_sha256(out / f) for name, f in (("partition", "s1_partition.npy"),
                                                                         ("tune_s1", "tune_s1.npy"), ("b_rest_s1", "b_rest_s1.npy"))},
        "outputs": {"manifest_sha256": file_sha256(arena_out / "manifest.json"), **manifest["sha256"],
                    "label_sha256": file_sha256(label_out / "label.npy"),
                    "true_count_sha256": file_sha256(label_out / "true_count.npy"),
                    "rec_owner_sha256": file_sha256(label_out / "rec_owner.npy")},
        "examples": ex,
        "caveats": ["real FR/TU pairs carry the arena v1 tree/blend of the US/India original (proxy, not "
                    "recomputed on French text); synthetic pairs carry tree 0.5 and blend --syn-blend",
                    "C2 step 40k was trained on partition A, so TU S1s are in-sample for the base model's "
                    "US/India text (not for the French rewrite)",
                    "département tokens are assigned at random (no geography learned; option A)"],
    }
    _json(out / "build_report.json", report)
    log(json.dumps(counts))
    # typed-negative composition exactly as fast_trainer computes it (train pairs only)
    from ber.fast_trainer import NegativeType, classify_negative_types
    t0 = time.monotonic()
    flags = classify_negative_types(reference, pool, left, right, label, tree, blend, owner_new)
    flags[~train] = 0
    gated = train & ~label & (tree >= args.tree_gate) & (flags == 0)
    comp = {kind_.name: {"french": int((((flags & int(kind_)) != 0) & fr_rows).sum()),
                         "replay": int((((flags & int(kind_)) != 0) & (world == 0)).sum())} for kind_ in NegativeType}
    comp["gated_random"] = {"french": int((gated & fr_rows).sum()), "replay": int((gated & (world == 0)).sum())}
    report["train_negative_composition"] = comp
    report["train_negative_composition_seconds"] = round(time.monotonic() - t0, 1)
    _json(out / "build_report.json", report)
    log(json.dumps(comp))
    return report


# ---------------------------------------------------------------------------
# Evaluation (B_REST gate + Frenchified tuning world)
# ---------------------------------------------------------------------------

def evaluate_cmd(args) -> dict:
    from ber.arena_decision import evaluate
    from ber.pair_tree_arena import decide, false_merge

    out = Path(args.output)
    pairs = pq.read_table(out / "arena/pairs.parquet", columns=["left_idx", "right_idx", "kind"])
    left = pairs.column(0).to_numpy().astype(np.int64)
    right = pairs.column(1).to_numpy().astype(np.int64)
    kind = pairs.column(2).to_numpy()
    label = np.load(out / "labels/label.npy")
    true_count = np.load(out / "labels/true_count.npy")
    partition = np.load(out / "s1_partition.npy")
    country = pq.read_table(out / "arena/reference.parquet", columns=["origin_country"]).column(0).to_numpy(zero_copy_only=False)
    tree = np.load(out / "arena/v1_tree.npy")
    n_s1 = len(true_count)
    scores = {}
    for name, directory in (("candidate", args.scores), ("baseline", args.baseline_scores)):
        if directory:
            s = np.nan_to_num(np.load(Path(directory) / "scores.npy").astype(np.float32))
            if s.shape != label.shape:
                raise ValueError(f"{name} scores are not aligned")
            scores[name] = s
    report = {"note": "B_REST = real US/India labels; TUNE = Frenchified train-derived world (labels by construction; "
                      "tree is a proxy = arena v1 tree of the original pair, 0.5 for synthetic pairs). Not French accuracy."}

    def run(prob, population, threshold):
        rows = population[left]
        accept = np.zeros(len(left), bool)
        accept[rows] = decide(prob[rows], left[rows], right[rows], threshold, n_s1)
        res = evaluate(accept, left, label, true_count, population=population, country=country, return_per_s1=True)
        per = np.asarray(res.pop("per_s1"))[population]
        res["false_merge"] = false_merge(accept, left, population, true_count)
        res["mean_list_length"] = float(np.bincount(left[accept], minlength=n_s1)[population].mean())
        return res, per, accept

    # B_REST gate: plain blend 0.9*C2 + 0.1*tree >= 0.10, tree gate 0.01
    pop = partition == CODE_B_REST
    block, per = {}, {}
    for name, s in scores.items():
        prob = np.where(tree >= args.gate, args.weight * s + (1 - args.weight) * tree, 0).astype(np.float32)
        block[name], per[name], _ = run(prob, pop, args.threshold)
    if len(per) == 2:
        d = per["candidate"] - per["baseline"]
        block["candidate_minus_baseline"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
        block["gate2_pass"] = bool(d.mean() >= -0.0005)
    report["B_REST"] = block

    # Tuning world
    pop = partition == CODE_SLICE
    tblock = {}
    for name, s in scores.items():
        entry = {}
        prob5 = np.where(tree >= args.gate, args.weight * s + (1 - args.weight) * tree, 0).astype(np.float32)
        res, per5, acc = run(prob5, pop, args.threshold)
        entry["rule5_0.9c2_0.1proxytree_ge_0.10"] = res
        rows_t = pop[left]
        n_t = int(pop.sum())
        entry["label_free_signatures"] = {
            "uncertain_band_per_s1_blend_0.05_0.5": float((rows_t & (prob5 >= .05) & (prob5 < .5)).sum() / n_t),
            "tree_high_c2_low_per_s1_proxy_tree": float((rows_t & (tree >= .9) & (s < .05)).sum() / n_t),
            "mean_list_len": res["mean_list_length"],
            "france_reference_c2_40k": {"uncertain_band_per_s1": 1.94, "tree_high_c2_low_per_s1": .545, "mean_list": 3.26}}
        entry["band_truth_rate_rule5"] = {
            "uncertain_0.05_0.5": float(label[rows_t & (prob5 >= .05) & (prob5 < .5)].mean()),
            "accepted_ge_0.7": float(label[rows_t & acc & (prob5 >= .7)].mean())}
        sweep = {}
        for t in args.sweep:
            r, _, _ = run(prob5, pop, t)
            sweep[f"{t:.2f}"] = {k: r[k] for k in ("macro_f05", "precision", "recall", "false_links", "mean_list_length")}
        entry["rule5_threshold_sweep"] = sweep
        entry["rule5_best_threshold"] = max(sweep, key=lambda k: sweep[k]["macro_f05"])
        sweep_c2 = {}
        prob_c2 = np.where(tree >= args.gate, s, 0).astype(np.float32)
        for t in args.sweep:
            r, _, _ = run(prob_c2, pop, t)
            sweep_c2[f"{t:.2f}"] = {k: r[k] for k in ("macro_f05", "precision", "recall", "false_links", "mean_list_length")}
        entry["c2_only_threshold_sweep"] = sweep_c2
        entry["c2_only_best_threshold"] = max(sweep_c2, key=lambda k: sweep_c2[k]["macro_f05"])
        rows = pop[left]
        entry["mean_score_by_kind"] = {KIND_NAMES[k]: float(s[rows & (kind == k)].mean()) if (rows & (kind == k)).any() else None
                                       for k in range(len(KIND_NAMES))}
        entry["accept_rate_by_kind_rule5"] = {KIND_NAMES[k]: float(acc[rows & (kind == k)].mean()) if (rows & (kind == k)).any() else None
                                              for k in range(len(KIND_NAMES))}
        tblock[name] = entry
        per[f"tune_{name}"] = per5
    if "tune_candidate" in per and "tune_baseline" in per:
        d = per["tune_candidate"] - per["tune_baseline"]
        tblock["candidate_minus_baseline_rule5"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
    report["TUNE_frenchified"] = tblock

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    report = clean(report)
    _json(Path(args.report), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "evaluate"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/az-ml-new/run27/S/arena_fr"))
    parser.add_argument("--arena", type=Path, default=Path("/workspace/az-ml-new/arena/v1"))
    parser.add_argument("--labels", type=Path, default=Path("/workspace/az-ml-new/arena/labels"))
    parser.add_argument("--arena-partition", type=Path, default=Path("/workspace/az-ml-new/arena/finalize_v1/s1_partition.npy"))
    parser.add_argument("--sealed-reservation", type=Path, default=Path("/workspace/az-ml-new/v2_holdout2"))
    parser.add_argument("--monitor-from", type=Path, default=Path("/workspace/az-ml-new/france_mix"))
    parser.add_argument("--vocab-counts", type=Path, default=Path("/workspace/az-ml-new/run27/S/vocab_probe.json"))
    parser.add_argument("--french-s1", type=int, default=120_000)
    parser.add_argument("--tune-s1", type=int, default=20_000)
    parser.add_argument("--french-share", type=float, default=.48)
    parser.add_argument("--p-legal-swap", type=float, default=.6)
    parser.add_argument("--p-decoy-word", type=float, default=.6)
    parser.add_argument("--p-sibling", type=float, default=.12)
    parser.add_argument("--p-legal-positive", type=float, default=.3)
    parser.add_argument("--tree-gate", type=float, default=.01)
    parser.add_argument("--syn-blend", type=float, default=.25,
                        help="v1-blend proxy of synthetic pairs (< 0.3: random-gated, not typed, negatives)")
    parser.add_argument("--fr-untyped-negative-keep", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--scores")
    parser.add_argument("--baseline-scores")
    parser.add_argument("--weight", type=float, default=.9)
    parser.add_argument("--threshold", type=float, default=.10)
    parser.add_argument("--gate", type=float, default=.01)
    parser.add_argument("--sweep", type=float, nargs="*", default=[.05, .1, .15, .2, .3, .4, .5, .6, .7])
    parser.add_argument("--report")
    args = parser.parse_args()
    if not 0 < args.french_share <= .5:
        parser.error("--french-share must be in (0, 0.5]")
    result = build(args) if args.command == "build" else evaluate_cmd(args)
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False)[:6000], flush=True)


if __name__ == "__main__":
    main()
