# Usage
# python generate_synthetic_corpus.py --outdir (where to save) --per-type (documents per template)
# --seed (for reproducibility) --profile (low, med, high prevalence PII), --probs (optional json with PII probs)

import os, json, random, argparse, re
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

# Labels and lists of possible PII
# Update for local requirements

FIRST_NAMES = ["Amelia","Olivia","Isla","Ava","Mia","Noah","Oliver","George","Arthur","Leo"]
LAST_NAMES = ["Smith","Jones","Taylor","Brown","Williams","Wilson","Johnson","Davies","Patel","Hussain"]
HOSPITALS = ["StHere Hospital","StThere Hospital","Central Orthopaedic Centre","StElsewhere General Hospital"]
UNITS = ["Trauma Unit","Radiology Department","Histopathology Laboratory","Upper Limb Service", "Trauma unit", "Oncology"]
EXT_HCO = ["Autumntown Health Centre","Summer Road Surgery","Spring Medical Practice","Winter Counties Imaging Ltd"]
PROFESSIONS = ["Consultant Orthopaedic Surgeon","Consultant Radiologist","Specialty Registrar","Consultant Pathologist"]
EPONYMOUS = ["Monteggia", "Galeazzi", "Bennett", "Colles"]
STREETS = ["North Street","East Road","West Road","South Road"]
CITIES = ["London","New York","Paris","Kuala Lumpur"]

LABELS = [
    "NAME_HCP","NAME_PATIENT",
    "ADDRESS_HOUSE_NUMBER","ADDRESS_STREET","ADDRESS_CITY","ADDRESS_POSTCODE",
    "DATE","AGE_OVER_89","PHONE","EMAIL","MRN","NHS_NUMBER","SPECIMEN_ID",
    "LICENSE_GMC","URL","HOSPITAL_UNIT","EXTERNAL_HCO","PROFESSIONAL_DETAILS","ACCOUNT"
]

# ---------------------------------------------------------------------
# Core builder that guarantees correct offsets for every entity
# ---------------------------------------------------------------------

class DocBuilder:
    def __init__(self):
        self._parts = []
        self._entities = []
        self._length = 0

    @property
    def length(self):
        return self._length

    @property
    def entities(self):
        return self._entities

    def add(self, s: str):
        """Append literal text without an entity."""
        if not s:
            return
        self._parts.append(s)
        self._length += len(s)

    def add_ent(self, s: str, label: str):
        """Append text and record its entity span against the global buffer."""
        start = self._length
        self._parts.append(s)
        end = start + len(s)
        self._entities.append({"start_offset": start, "end_offset": end, "label": label})
        self._length = end

    def text(self) -> str:
        return "".join(self._parts)

def random_date(a=2020, b=2022):
    start = datetime(a,1,1); end = datetime(b,12,31)
    d = start + (end-start) * random.random()
    return d.strftime(random.choice(["%d/%m/%Y","%Y-%m-%d","%d %b %Y"]))

def name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def hcp_name():
    titles = ["Dr","Mr","Ms","Prof"]
    return f"{random.choice(titles)} {name()}"

def nhs_num():
    def cd(nine):
        w = list(range(10,1,-1))
        s = sum(int(d)*wi for d,wi in zip(nine,w)); r = s%11; c = 11-r
        if c==11: return "0"
        if c==10: return None
        return str(c)
    while True:
        nine = "".join(str(random.randint(0,9)) for _ in range(9))
        c = cd(nine)
        if c:
            return f"{nine[:3]} {nine[3:6]} {nine[6:9]} {c}" if random.random()<0.5 else nine+c

def mrn():
    return f"JR{random.randint(100000,999999)}"

def phone():
    return random.choice([
        f"01865 {random.randint(200000,299999)}",
        f"07{random.randint(400000000,999999999)}",
        f"Ext {random.randint(10000,99999)}",
        f"Bleep {random.randint(1000,9999)}"
    ])

def specimen():
    return f"{random.randint(20,24)}-HS-{random.randint(10000,99999)}"

def gmc():
    return str(random.randint(1000000,9999999))

def address_parts():
    num = str(random.randint(1,250)); street = random.choice(STREETS); city = random.choice(CITIES)
    outward = random.choice(["OX","SN","RG"]) + str(random.randint(1,9))
    inward = f"{random.randint(0,9)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
    return num, street, city, f"{outward} {inward}"

def redact(text, entities):
    out, last = [], 0
    for e in sorted(entities, key=lambda x:x["start_offset"]):
        out.append(text[last:e["start_offset"]]); out.append(f"[{e['label']}]"); last = e["end_offset"]
    out.append(text[last:])
    return "".join(out)

def count_words(s): 
    return len(re.findall(r"\b\w+\b", s))

# Helper to build a line made of optional fields with a chosen separator
# Each field is defined as a list of segments where a segment is either:
#   ("lit", text) or ("ent", text, label)
def flush_fields(builder: DocBuilder, fields, sep=" | ", end="\n"):
    present = [fld for fld in fields if fld]  # keep only non empty fields
    for i, fld in enumerate(present):
        if i > 0:
            builder.add(sep)
        for seg in fld:
            if seg[0] == "lit":
                builder.add(seg[1])
            elif seg[0] == "ent":
                builder.add_ent(seg[1], seg[2])
            else:
                raise ValueError("Unknown segment type")
    if present and end:
        builder.add(end)

# Sentence templates

def finding_sentence_radiology():
    ep = random.choice(EPONYMOUS)
    options = [
        f"No acute disease. No features of {ep} fracture.",
        f"There is evidence of {ep} fracture.",
        "No abnormalities seen."
    ]
    return random.choice(options)

def finding_sentence_msk():
    ep = random.choice(EPONYMOUS)
    options = [
        f"No acute osseous injury. No evidence of {ep} pattern.",
        f"There is evidence of {ep} pattern.",
        "There is a small joint effusion.",
        "Evidence of osteoarthritis."
    ]
    return random.choice(options)

def diagnosis_sentence_histo():
    return random.choice(["Diagnosis: Benign.","Diagnosis: Malignant."])

def radiology(p):
    b = DocBuilder()

    # Header with optional hospital and unit
    header_fields = []
    if random.random() < p["HOSPITAL_UNIT"]:
        header_fields.append([("ent", random.choice(HOSPITALS), "HOSPITAL_UNIT")])
    if random.random() < p["HOSPITAL_UNIT"]:
        header_fields.append([("lit","Unit: "), ("ent", random.choice(UNITS), "HOSPITAL_UNIT")])
    flush_fields(b, header_fields, sep=" | ", end="\n")

    # Date, NHS, MRN
    line2_fields = []
    if random.random() < p["DATE"]:
        line2_fields.append([("lit","Date: "), ("ent", random_date(), "DATE")])
    if random.random() < p["NHS_NUMBER"]:
        line2_fields.append([("lit","NHS No: "), ("ent", nhs_num(), "NHS_NUMBER")])
    if random.random() < p["MRN"]:
        line2_fields.append([("lit","Hospital No: "), ("ent", mrn(), "MRN")])
    flush_fields(b, line2_fields, sep=" | ", end="\n")

    # Patient and DOB
    line3_fields = []
    if random.random() < p["NAME_PATIENT"]:
        line3_fields.append([("lit","Patient: "), ("ent", name(), "NAME_PATIENT")])
    if random.random() < p["DATE"]:
        line3_fields.append([("lit","DOB: "), ("ent", random_date(1930,1985), "DATE")])
    flush_fields(b, line3_fields, sep=" | ", end="\n")

    # Referrer line
    parts_ref = []
    have_name = random.random() < p["NAME_HCP"]
    have_role = random.random() < p["PROFESSIONAL_DETAILS"]
    have_phone = random.random() < p["PHONE"]
    have_email = random.random() < p["EMAIL"]

    if have_name:
        fld = [("lit","Referrer: "), ("ent", hcp_name(), "NAME_HCP")]
        if have_role:
            fld += [("lit"," ("), ("ent", random.choice(PROFESSIONS), "PROFESSIONAL_DETAILS"), ("lit",")")]
        parts_ref.append(fld)
    else:
        if have_role:
            parts_ref.append([("lit","Referrer: ("), ("ent", random.choice(PROFESSIONS), "PROFESSIONAL_DETAILS"), ("lit",")")])
    if have_phone:
        parts_ref.append([("lit","Contact "), ("ent", phone(), "PHONE")])
    if have_email:
        parts_ref.append([("ent", name().lower().replace(" ",".")+"@nhs.net", "EMAIL")])
    flush_fields(b, parts_ref, sep=" | ", end="\n")

    # Address only if present
    if random.random() < p["ADDRESS"]:
        n, st, city, pc = address_parts()
        b.add("Address: ")
        b.add_ent(n, "ADDRESS_HOUSE_NUMBER"); b.add(" ")
        b.add_ent(st, "ADDRESS_STREET"); b.add(", ")
        b.add_ent(city, "ADDRESS_CITY"); b.add(" ")
        b.add_ent(pc, "ADDRESS_POSTCODE")
        b.add("\n")

    # Findings and conclusion
    b.add("\nExamination: Chest X-ray\n")
    b.add("Findings: "); b.add(finding_sentence_radiology()); b.add("\n")
    b.add("Conclusion: Unremarkable.\n")

    # Reporter line
    want_name = random.random() < p["NAME_HCP"]
    want_gmc = (random.random() < p["LICENSE_GMC"]) if want_name else False
    if want_name:
        b.add("Reported by ")
        b.add_ent(hcp_name(), "NAME_HCP")
        if want_gmc:
            b.add(", GMC ")
            b.add_ent(gmc(), "LICENSE_GMC")
        b.add(".\n")

    return b.text(), b.entities, "general_radiology"

def histo(p):
    b = DocBuilder()

    # Laboratory if unit present
    if random.random() < p["HOSPITAL_UNIT"]:
        b.add("Laboratory: ")
        b.add_ent(random.choice(UNITS), "HOSPITAL_UNIT")
        b.add("\n")

    # Specimen line
    fields = []
    if random.random() < p["SPECIMEN_ID"]:
        fields.append([("lit","Specimen Identifier: "), ("ent", specimen(), "SPECIMEN_ID")])
    if random.random() < p["NAME_PATIENT"]:
        fields.append([("lit","Patient: "), ("ent", name(), "NAME_PATIENT")])
    if random.random() < p["MRN"]:
        fields.append([("lit","Hospital No: "), ("ent", mrn(), "MRN")])
    if random.random() < p["NHS_NUMBER"]:
        fields.append([("lit","NHS No: "), ("ent", nhs_num(), "NHS_NUMBER")])
    flush_fields(b, fields, sep=" | ", end="\n")

    # Diagnosis
    b.add(diagnosis_sentence_histo()); b.add("\n")

    # Authorised by
    want_name = random.random() < p["NAME_HCP"]
    want_role = random.random() < p["PROFESSIONAL_DETAILS"]
    want_date = random.random() < p["DATE"]

    if want_name or want_role or want_date:
        b.add("Authorised by ")
        if want_name:
            b.add_ent(hcp_name(), "NAME_HCP")
        if want_role:
            b.add(" ("); b.add_ent("Consultant Pathologist", "PROFESSIONAL_DETAILS"); b.add(")")
        if want_date:
            if want_name or want_role:
                b.add(" on ")
            b.add_ent(random_date(), "DATE")
        b.add(".\n")

    return b.text(), b.entities, "histopathology"

def msk(p):
    b = DocBuilder()

    # Site and service
    if random.random() < p["HOSPITAL_UNIT"]:
        b.add("Site: ")
        b.add_ent(random.choice(HOSPITALS), "HOSPITAL_UNIT")
        b.add(" / Service: Upper Limb Service\n")
    else:
        b.add("Service: Upper Limb Service\n")

    # Requesting clinician, date, contact
    parts = []
    want_name = random.random() < p["NAME_HCP"]
    want_role = random.random() < p["PROFESSIONAL_DETAILS"]

    if want_name or want_role:
        fld = [("lit","Requesting clinician: ")]
        if want_name:
            fld += [("ent", hcp_name(), "NAME_HCP")]
        if want_role:
            if want_name:
                fld += [("lit"," (")]
            else:
                fld += [("lit","(")]
            fld += [("ent", random.choice(PROFESSIONS), "PROFESSIONAL_DETAILS"), ("lit",")")]
        parts.append(fld)

    if random.random() < p["DATE"]:
        parts.append([("lit","Date: "), ("ent", random_date(), "DATE")])

    if random.random() < p["PHONE"]:
        parts.append([("lit","Contact: "), ("ent", phone(), "PHONE")])

    flush_fields(b, parts, sep=" | ", end="\n")

    # External organisation
    if random.random() < p["EXTERNAL_HCO"]:
        b.add("Organisation: "); b.add_ent(random.choice(EXT_HCO), "EXTERNAL_HCO"); b.add("\n")

    # Findings
    b.add("Investigation: XR Wrist.\n")
    b.add("Findings: "); b.add(finding_sentence_msk()); b.add("\n")

    # Account number if present
    if random.random() < p["ACCOUNT"]:
        b.add("Accession: "); b.add_ent(f"ACC{random.randint(100000,999999)}", "ACCOUNT"); b.add(".\n")

    return b.text(), b.entities, "msk_imaging"

def generate(outdir, per_type, seed, profile, probs_path):
    random.seed(seed)
    if profile == "low":
        p = {"HOSPITAL_UNIT":0.2,"DATE":0.15,"NHS_NUMBER":0.02,"MRN":0.03,"NAME_PATIENT":0.02,"NAME_HCP":0.08,
             "PROFESSIONAL_DETAILS":0.05,"PHONE":0.05,"EMAIL":0.02,"ADDRESS":0.03,"LICENSE_GMC":0.03,
             "SPECIMEN_ID":0.03,"EXTERNAL_HCO":0.03,"ACCOUNT":0.03}
    elif profile == "med":
        p = {"HOSPITAL_UNIT":0.9,"DATE":0.7,"NHS_NUMBER":0.4,"MRN":0.4,"NAME_PATIENT":0.25,"NAME_HCP":0.6,
             "PROFESSIONAL_DETAILS":0.6,"PHONE":0.5,"EMAIL":0.3,"ADDRESS":0.3,"LICENSE_GMC":0.6,"SPECIMEN_ID":0.5,
             "EXTERNAL_HCO":0.5,"ACCOUNT":0.4}
    else:
        p = {"HOSPITAL_UNIT":1.0,"DATE":0.9,"NHS_NUMBER":0.8,"MRN":0.8,"NAME_PATIENT":0.7,"NAME_HCP":0.9,
             "PROFESSIONAL_DETAILS":0.9,"PHONE":0.8,"EMAIL":0.6,"ADDRESS":0.6,"LICENSE_GMC":0.9,"SPECIMEN_ID":0.8,
             "EXTERNAL_HCO":0.7,"ACCOUNT":0.7}
    if probs_path:
        with open(probs_path,"r",encoding="utf-8") as f:
            p = json.load(f)

    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    gens = [radiology, histo, msk]
    docs = []; label_counts = Counter(); pii_char=0; total_char=0; pii_words=0; total_words=0; did=0
    for g in gens:
        for _ in range(per_type):
            text, ents, dtype = g(p)
            red = redact(text, ents)
            docs.append({"id": f"syn_{did:04d}", "doc_type": dtype, "text": text, "entities": ents, "redacted_text": red}); did += 1
            total_char += len(text); total_words += count_words(text)
            for e in ents:
                label_counts[e["label"]] += 1
                pii_char += (e["end_offset"] - e["start_offset"])
                pii_words += count_words(text[e["start_offset"]:e["end_offset"]])

    with open(out/"synthetic_corpus.jsonl","w",encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"id": d["id"], "doc_type": d["doc_type"], "text": d["text"], "entities": d["entities"]}, ensure_ascii=False) + "\n")
    with open(out/"synthetic_corpus_redacted.jsonl","w",encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"id": d["id"], "doc_type": d["doc_type"], "redacted_text": d["redacted_text"]}, ensure_ascii=False) + "\n")
    with open(out/"labels.json","w",encoding="utf-8") as f:
        json.dump({"labels": LABELS}, f, indent=2)

    stats = {
        "num_documents": len(docs),
        "documents_per_type": dict(Counter(d["doc_type"] for d in docs)),
        "label_counts": dict(label_counts),
        "approx_pii_word_fraction": round(pii_words / max(total_words,1), 4),
        "approx_pii_char_fraction": round(pii_char / max(total_char,1), 4),
        "seed": seed, "per_type": per_type, "profile": profile, "probs": p
    }
    with open(out/"STATS.json","w",encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

# Argparse
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="synthetic_data")
    ap.add_argument("--per-type", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--profile", type=str, choices=["low","med","high"], default="low")
    ap.add_argument("--probs", type=str, default="", help="Optional JSON file with inclusion probabilities")
    args = ap.parse_args()
    generate(args.outdir, args.per_type, args.seed, args.profile, args.probs)
