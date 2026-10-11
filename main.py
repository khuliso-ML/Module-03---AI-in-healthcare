import streamlit as st
st.title("Hi, welcome to my app")

"""DiagnoSA - Streamlit test harness.

RESEARCH / DEMO ONLY. Not a medical device. Not for clinical use.
Run:  streamlit run main.py
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms

    TORCH_OK = True
except ImportError:
    TORCH_OK = False

WEIGHTS = Path("models/diagnosa_densenet121.pt")  # state_dict, 8 outputs
LABELS = ["TB", "Pneumonia", "Pleural effusion", "Cardiomegaly",
          "Consolidation", "Edema", "Pneumothorax", "Nodule/Mass"]
# TB gets a lower decision threshold (sensitivity favoured, mirrors upweighting in training)
THRESH = {l: 0.50 for l in LABELS}
THRESH["TB"] = 0.35

SYMPTOMS = {
    "Cough > 2 weeks": 1, "Fever": 1, "Night sweats": 1,
    "Weight loss": 1, "TB contact": 1, "Coughing blood": 2,
}


# ---------------------------------------------------------------- CXR module
@st.cache_resource
def load_model():
    """Returns a DenseNet-121 if torch + weights exist, else None (demo mode)."""
    if not (TORCH_OK and WEIGHTS.exists()):
        return None
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, len(LABELS))
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return m.eval()


def preprocess(img):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(img.convert("RGB")).unsqueeze(0)


def run_cxr(img, raw, model):
    """Returns (probs dict, gradcam array or None)."""
    if model is None:  # demo mode: deterministic fake scores from image hash
        seed = int(hashlib.sha256(raw).hexdigest(), 16) % (2**32)
        p = np.random.default_rng(seed).beta(1.2, 4, len(LABELS))
        return dict(zip(LABELS, p.round(3))), None

    x = preprocess(img)
    feats = model.features(x)
    feats.retain_grad()
    logits = model.classifier(F.adaptive_avg_pool2d(F.relu(feats), 1).flatten(1))
    probs = torch.sigmoid(logits)[0].detach().numpy()
    model.zero_grad()
    logits[0, int(probs.argmax())].backward()  # Grad-CAM on top class
    w = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats).sum(1)).squeeze().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return dict(zip(LABELS, probs.round(3))), cam


def overlay(img, cam):
    base = img.convert("RGB")
    h = np.array(Image.fromarray((cam * 255).astype("uint8")).resize(base.size, Image.BILINEAR)) / 255.0
    out = np.array(base) * (1 - 0.5 * h[..., None]) + np.array([255, 0, 0]) * 0.5 * h[..., None]
    return Image.fromarray(out.astype("uint8"))


# -------------------------------------------------------------- Smart Triage
def triage(probs, symptoms, spo2, hiv, age):
    score = round(probs["TB"] * 4)
    score += sum(SYMPTOMS[s] for s in symptoms)
    score += 4 if spo2 < 92 else 2 if spo2 < 95 else 0
    score += 1 if hiv == "Positive" else 0
    score += 1 if age < 5 or age > 65 else 0
    score += 4 if probs["Pneumothorax"] >= THRESH["Pneumothorax"] else 0
    level = "Urgent" if score >= 7 else "Priority" if score >= 4 else "Routine"
    return level, score


# ----------------------------------------------------------------- EMR Bridge
def to_fhir(r):
    """Simplified FHIR-style DiagnosticReport (illustrative, not validated)."""
    return {
        "resourceType": "DiagnosticReport",
        "id": r["id"],
        "status": "preliminary",
        "code": {"text": "AI-assisted chest X-ray screening (decision support)"},
        "subject": {"identifier": {"value": r["id"]}},
        "effectiveDateTime": r["time"],
        "conclusion": f"Top finding: {r['top_finding']}; TB probability {r['tb_prob']}; "
                      f"triage {r['level']}. Requires clinician review.",
        "extension": [{"url": "triage-score", "valueInteger": r["score"]},
                      {"url": "inference-mode", "valueString": r["mode"]}],
    }


# ------------------------------------------------------------------------ UI
st.set_page_config(page_title="DiagnoSA", page_icon="🩻", layout="wide")
st.session_state.setdefault("records", [])
st.session_state.setdefault("last", None)

model = load_model()
mode = "model" if model is not None else "demo"

with st.sidebar:
    st.title("🩻 DiagnoSA")
    offline = st.toggle("Offline mode", help="Queues records locally; EMR sync disabled.")
    st.caption("Inference mode: **" + ("trained model" if model else "DEMO (simulated scores)") + "**")
    if model is None:
        st.warning(f"No weights at `{WEIGHTS}`. Scores are simulated and meaningless clinically.")
    st.error("Research prototype. Not a medical device. Clinician must review every result.")

tab1, tab2, tab3, tab4 = st.tabs(["CXR Screener + Triage", "Clinic Dashboard", "EMR Bridge", "Compliance notes"])

with tab1:
    consent = st.checkbox("Patient informed of AI use and consent obtained (POPIA)")
    up = st.file_uploader("Chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 0, 110, 35)
    spo2 = c2.number_input("SpO₂ (%)", 50, 100, 96)
    hiv = c3.selectbox("HIV status", ["Unknown", "Negative", "Positive"])
    syms = st.multiselect("Symptoms", list(SYMPTOMS))

    if st.button("Analyse", type="primary", disabled=not (consent and up)):
        raw = up.getvalue()
        img = Image.open(up)
        probs, cam = run_cxr(img, raw, model)
        level, score = triage(probs, syms, spo2, hiv, age)
        top = max(probs, key=probs.get)
        rec = {"id": "PT-" + uuid.uuid4().hex[:8],  # pseudonymous, no names stored
               "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "age": age, "hiv": hiv, "spo2": spo2, "symptoms": ", ".join(syms),
               "top_finding": top, "tb_prob": float(probs["TB"]), "level": level,
               "score": score, "mode": mode, "synced": False}
        st.session_state.records.append(rec)
        st.session_state.last = (img, probs, cam, rec)

    if st.session_state.last:
        img, probs, cam, rec = st.session_state.last
        icon = {"Urgent": "🔴", "Priority": "🟠", "Routine": "🟢"}[rec["level"]]
        st.subheader(f"{icon} Triage: {rec['level']} (score {rec['score']})  ·  {rec['id']}")
        a, b = st.columns(2)
        a.image(overlay(img, cam) if cam is not None else img,
                caption="Grad-CAM (top class)" if cam is not None else "Uploaded image (no heatmap in demo mode)")
        df = pd.DataFrame({"Finding": probs.keys(), "Probability": probs.values(),
                           "Flagged": [p >= THRESH[k] for k, p in probs.items()]})
        b.dataframe(df, hide_index=True, width="stretch",
                    column_config={"Probability": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0)})
        b.caption(f"TB flag threshold {THRESH['TB']} (others {THRESH['Pneumonia']}). "
                  "Lower TB threshold favours sensitivity and increases false positives.")

with tab2:
    recs = pd.DataFrame(st.session_state.records)
    if recs.empty:
        st.info("No cases analysed yet this session.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Cases", len(recs))
        m2.metric("Urgent", int((recs.level == "Urgent").sum()))
        m3.metric("TB flagged", int((recs.tb_prob >= THRESH["TB"]).sum()))
        st.bar_chart(recs.level.value_counts())
        st.dataframe(recs, hide_index=True, width="stretch")
        st.download_button("Download CSV", recs.to_csv(index=False), "diagnosa_cases.csv")

with tab3:
    pending = [r for r in st.session_state.records if not r["synced"]]
    st.write(f"**Queued for sync:** {len(pending)}  ·  **Network:** {'offline' if offline else 'online'}")
    if st.button("Sync to EMR (mock)", disabled=offline or not pending):
        for r in pending:
            r["synced"] = True
        st.success(f"Synced {len(pending)} record(s) to mock EMR.")
    if st.session_state.records:
        st.caption("Latest FHIR-style payload")
        st.json(to_fhir(st.session_state.records[-1]))
        st.download_button("Download JSON",
                           json.dumps([to_fhir(r) for r in st.session_state.records], indent=2),
                           "diagnosa_fhir.json")

with tab4:
    st.markdown("""
**Built into this prototype**
- Consent gate before analysis (POPIA)
- Pseudonymous IDs only; no patient names or direct identifiers
- Human-in-the-loop messaging; outputs are decision support
- Explainability via Grad-CAM (trained-model mode)
- Per-class thresholds with documented TB sensitivity trade-off
- Inference mode and timestamps recorded for audit

**Still needed before any real use**
- Validation on South African clinical data, with subgroup performance (age, sex, HIV status, equipment)
- SAHPRA registration; FDA 510(k)/De Novo if targeting the US
- Encryption at rest and in transit, access control, audit logging
- ISO 13485, IEC 62304, ISO 14971 documentation
- Ethics committee approval for any study with patient data
- Post-deployment monitoring and incident process
""")