import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time
from io import StringIO

try:
    import faiss
    _have_faiss = True
except Exception:
    _have_faiss = False

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoder
except Exception as e:
    st.error("Please install sentence-transformers.")
    raise e

try:
    from rank_bm25 import BM25Okapi
except Exception:
    st.error("Please install rank_bm25.")
    raise

try:
    import google.generativeai as genai
    _have_genai = True
except Exception:
    genai = None
    _have_genai = False

# Config 
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_WEIGHTS = {
    "skills": 0.50,
    "availability": 0.20,
    "experience": 0.10,
    "rating": 0.08,
    "cost": 0.07,
    "certifications": 0.05
}
SAMPLE_CSV = """Resource ID,Resource Name,Delivery Manager,Current Status,Primary Skill,Detailed Skill Set,Remarks,Category,Secondary Skill,Skill Source,DS Allocation Percentage,Identified to GTP,Core Resource,Old Remarks,DATA-TP Start Date,Ageing (in days),Account Feedback,Resource Rating (1-5),Interview Rejection Count,Band,Total Experience (Years),Company Experience (Years),Location Type,Employment Type,Employee Status,Current Location,Standard Cost,Phone Number,Certifications,Approval Status,Days in Pool
R001,Anil Kumar,DM-A,Bench,Python,"python, spark, aws, sql","Experienced in ETL and streaming",Dev,Scala,Internal,10,No,Yes,,2024-07-01,120,Good,4,0,B2,6,3,Onsite,Full-time,Active,Bengaluru,120000,9999999999,"AWS Certified",Approved,120
R002,Geeta Singh,DM-B,Bench,Java,"java, spring, aws, microservices","Strong backend + cloud experience",Dev,SQL,Referral,0,No,Yes,,2024-09-15,60,Excellent,5,0,B3,8,5,Remote,Contract,Active,Pune,140000,8888888888,"OCJP",Approved,60
R003,Rajesh Iyer,DM-C,OnLeave,Python,"python, flask, gcp, docker","Mostly web apps, basic data work",Dev,Docker,Internal,50,Yes,No,,2024-03-01,300,Average,3,1,B1,4,2,Remote,Full-time,Active,Hyderabad,90000,7777777777,"GCP Associate",Pending,300
"""

# Utilities
def normalize_skill_text(s):
    if pd.isna(s) or s is None:
        return ""
    toks = [t.strip().lower() for t in re.split(r"[;,|\n]+", str(s)) if t.strip()]
    return ", ".join(sorted(set(toks)))

def preprocess_df(df):
    df = df.copy()
    expected = ["Resource ID","Resource Name","Primary Skill","Detailed Skill Set",
                "Certifications","Resource Rating","Total Experience (Years)",
                "DS Allocation Percentage","Standard Cost","Days in Pool",
                "Approval Status","Employee Status"]
    for c in expected:
        if c not in df.columns:
            df[c] = ""
    df["Primary Skill"] = df["Primary Skill"].fillna("").astype(str).str.strip().str.lower()
    df["Detailed Skill Set"] = df["Detailed Skill Set"].apply(normalize_skill_text)
    df["Certifications"] = df["Certifications"].fillna("").astype(str)
    df["Resource Rating"] = pd.to_numeric(df.get("Resource Rating", 0), errors="coerce").fillna(0)
    df["Total Experience (Years)"] = pd.to_numeric(df.get("Total Experience (Years)", 0), errors="coerce").fillna(0)
    df["DS Allocation Percentage"] = pd.to_numeric(df.get("DS Allocation Percentage", 0), errors="coerce").fillna(0)
    df["Standard Cost"] = pd.to_numeric(df.get("Standard Cost", 0), errors="coerce").fillna(0)
    df["Days in Pool"] = pd.to_numeric(df.get("Days in Pool", 9999), errors="coerce").fillna(9999)
    return df

def compose_doc_text(row):
    parts = [
        str(row.get("Primary Skill","")),
        str(row.get("Detailed Skill Set","")),
        str(row.get("Remarks","")),
        str(row.get("Certifications","")),
        str(row.get("Old Remarks","")),
        str(row.get("Account Feedback",""))
    ]
    return " . ".join([p for p in parts if p])

# Embedding & Index
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner=False)
def get_reranker_model():
    return CrossEncoder(RERANK_MODEL)

def build_index(docs_texts):
    emb_model = get_embedding_model()
    embeddings = emb_model.encode(docs_texts, convert_to_numpy=True, show_progress_bar=False)
    if _have_faiss:
        d = embeddings.shape[1]
        idx = faiss.IndexFlatL2(d)
        idx.add(embeddings)
        return idx, embeddings
    else:
        return None, embeddings

# BM25 
def build_bm25(docs_texts):
    tokenized = [t.split() for t in docs_texts]
    bm25 = BM25Okapi(tokenized)
    return bm25

# Hybrid search 
def vector_search(index, embeddings, query, top_k=50):
    emb_model = get_embedding_model()
    q_emb = emb_model.encode([query], convert_to_numpy=True)[0]
    if _have_faiss and index is not None:
        D,I = index.search(np.array([q_emb]), top_k)
        return list(I[0])
    else:
        q = q_emb / np.linalg.norm(q_emb) if np.linalg.norm(q_emb) > 0 else q_emb
        E = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sims = np.dot(E, q)
        ids = np.argsort(-sims)[:top_k]
        return ids.tolist()

def hybrid_search(query, docs_texts, index, embeddings, bm25=None, top_k_vector=50, top_k_bm25=30):
    vec_ids = vector_search(index, embeddings, query, top_k=top_k_vector)
    bm25_ids = []
    if bm25:
        tokenized = query.split()
        bm25_res = bm25.get_top_n(tokenized, docs_texts, n=top_k_bm25)
        bm25_ids = [docs_texts.index(x) for x in bm25_res]
    seen = set()
    combined = []
    for i in vec_ids + bm25_ids:
        if i not in seen:
            seen.add(i)
            combined.append(i)
    return combined

# Rerank 
def rerank_topk(query, docs_texts, candidate_indices, top_k=20):
    reranker = get_reranker_model()
    pairs = [(query, docs_texts[i]) for i in candidate_indices]
    scores = reranker.predict(pairs, show_progress_bar=False)
    scored = list(zip(candidate_indices, scores))
    scored_sorted = sorted(scored, key=lambda x: -x[1])
    top = [i for i,_ in scored_sorted[:top_k]]
    return top, scored_sorted

def extract_skills_llm(jd_text):
    """
    Uses Gemini to extract normalized technical skills from a free-text JD.
    Returns a list of lowercase skill names (strings).
    """

    key = get_gemini_key()
    if not key or not _have_genai:
        return []

    try:
        client = genai.Client(api_key=key)

        prompt = (
            "Extract ALL technical skills mentioned in the following job description. "
            "Return ONLY a JSON array of skill names in lowercase. No explanations.\n\n"
            f"JD:\n{jd_text}\n\n"
            "Output format example:\n[\"python\", \"spark\", \"aws\", \"docker\", \"mongodb\"]"
        )

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            config={"maxOutputTokens": 200}
        )

        text_out = resp.text() if callable(resp.text) else resp.text

        # Try to load JSON list
        import json
        skills = json.loads(text_out)

        # Normalize
        skills = [s.strip().lower() for s in skills if isinstance(s, str)]

        return skills

    except Exception as e:
        return []


def extract_from_jd(jd_text):
    jd = jd_text.lower()

    # 1. Extract skills using Gemini (Option D)
    extracted_skills = extract_skills_llm(jd_text)

    # 2. Pull out other structured items using regex (same as before)
    min_exp = 0
    m = re.search(r"(\d+)\+?\s*[-]?\s*years?", jd)
    if m:
        min_exp = int(m.group(1))

    max_days = 30 if "30" in jd or "immediate" in jd else 90

    budget = None
    m2 = re.search(r"<=\s*([0-9,]+)", jd)
    if m2:
        try:
            budget = int(m2.group(1).replace(",", ""))
        except:
            budget = None

    return {
        "required_skills": extracted_skills,
        "min_experience": min_exp,
        "max_days_to_start": max_days,
        "budget": budget
    }

#  Scoring 
def compute_feature_scores(query_info, meta):
    q_skills = set([s.strip().lower() for s in query_info.get("required_skills", []) if s])
    doc_skills = set([s.strip().lower() for s in str(meta.get("Detailed Skill Set","")).split(",") if s.strip()])
    skill_score = 0.0 if len(q_skills)==0 else len(q_skills & doc_skills)/len(q_skills)
    doc_exp = float(meta.get("Total Experience (Years)", 0.0))
    min_exp = float(query_info.get("min_experience", 0.0))
    exp_score = 1.0 if min_exp<=0 else min(1.0, doc_exp/min_exp)
    days = float(meta.get("Days in Pool", 9999))
    rating = float(meta.get("Resource Rating", 0.0))
    cost = float(meta.get("Standard Cost", 0.0))
    certs = 1.0 if str(meta.get("Certifications","")).strip() else 0.0
    return {"skills": skill_score, "experience": exp_score, "days_in_pool": days, "rating": rating, "cost": cost, "certifications": certs}

def normalize(x, minv, maxv):
    if maxv == minv:
        return 0.0
    return (x - minv) / (maxv - minv)

def compute_weighted_score(feature_scores, weights, days_min=0, days_max=365, cost_min=0, cost_max=1):
    days = feature_scores["days_in_pool"]
    avail = 1.0 - normalize(days, days_min, days_max)
    exp = feature_scores["experience"]
    skills = feature_scores["skills"]
    rating = normalize(feature_scores["rating"], 0, 5)
    cost_norm = normalize(feature_scores["cost"], cost_min, max(cost_max, cost_min+1))
    cost_score = 1.0 - cost_norm
    score = (weights["skills"] * skills +
             weights["availability"] * avail +
             weights["experience"] * exp +
             weights["rating"] * rating +
             weights["cost"] * cost_score +
             weights["certifications"] * feature_scores["certifications"])
    return float(max(0.0, min(1.0, score)))

# Gemini integration 
def get_gemini_key():
    try:
        k = st.secrets["GEMINI_API_KEY"]
        if k:
            return k
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", None)

def synthesize_with_gemini(candidate_meta, score, evidence,
                           gemini_model="gemini-2.5-flash", max_output_tokens=160):
    
    key = get_gemini_key()
    if not key:
        return None
    if not _have_genai:
        return None

    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(gemini_model)

        prompt = (
            "You are an assistant summarizing why a candidate is recommended for a job.\n"
            f"Candidate metadata: {candidate_meta}\n"
            f"Evidence: {evidence}\n"
            "Provide a concise 1–2 sentence justification referencing ONLY the evidence."
        )

        try:
            payload = {
                "content": {
                    "parts": [
                        {"type": "text", "text": prompt}
                    ]
                },
                "max_output_tokens": max_output_tokens
            }
            response = model.generate_content(payload)
        except Exception as e_a:
            try:
                payload = {
                    "content": [
                        {"type": "text", "text": prompt}
                    ],
                    "max_output_tokens": max_output_tokens
                }
                response = model.generate_content(payload)
            except Exception as e_b:
                try:
                    response = model.generate_content(prompt)
                except Exception as e_c:
                    return f"(Gemini call failed; tried multiple payloads) Errors: A:{e_a} | B:{e_b} | C:{e_c}"


        try:
            # 1
            if hasattr(response, "text") and response.text:
                return response.text.strip()
        except Exception:
            pass

        # 2
        try:

            if hasattr(response, "result") and response.result:
                r0 = response.result[0]
                if hasattr(r0, "content") and r0.content:
                    c0 = r0.content[0]
                    # c0 may have 'text' attribute
                    if hasattr(c0, "text") and c0.text:
                        return c0.text.strip()

            if hasattr(response, "output") and response.output:
                out = response.output

                try:
                    if isinstance(out, (list, tuple)) and len(out) > 0:
                        o0 = out[0]
                        if isinstance(o0, dict) and "content" in o0:
                            cont = o0["content"]
                            if isinstance(cont, list) and len(cont) > 0 and isinstance(cont[0], dict) and "text" in cont[0]:
                                return cont[0]["text"].strip()
                except Exception:
                    pass
        except Exception:
            pass

        # 3
        try:
            if hasattr(response, "candidates") and response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, "content") and cand.content:
                    pc = cand.content[0]
                    if hasattr(pc, "text") and pc.text:
                        return pc.text.strip()
        except Exception:
            pass

        # 4
        try:
            return str(response)[:2000]
        except Exception as e:
            return f"(Gemini: could not parse response; error: {e})"

    except Exception as top_e:
        return f"(Gemini error: {top_e})"

# Streamlit UI 
st.set_page_config(page_title="AI Resource Planning Assistant", layout="wide")
st.title("AI Resource Planning Assistant")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload resources CSV", type=["csv"])
    jd_text = st.text_area("Paste JD text here", height=220, value="""
We need 2 Backend Engineers for a 6-month engagement.
Required skills: Python, Spark, AWS.
Min experience: 4 years.
Availability: start within 30 days.
Location: Remote or India.
Budget: Standard Cost <= 150000 (monthly).
Approval: Approved resources only.
""")
    st.header("Scoring weights")
    w_skill = st.slider("Skills", 0.0, 1.0, DEFAULT_WEIGHTS["skills"])
    w_avail = st.slider("Availability", 0.0, 1.0, DEFAULT_WEIGHTS["availability"])
    w_exp = st.slider("Experience", 0.0, 1.0, DEFAULT_WEIGHTS["experience"])
    w_rating = st.slider("Rating", 0.0, 1.0, DEFAULT_WEIGHTS["rating"])
    w_cost = st.slider("Cost", 0.0, 1.0, DEFAULT_WEIGHTS["cost"])
    w_certs = st.slider("Certifications", 0.0, 1.0, DEFAULT_WEIGHTS["certifications"])
    weights = {"skills": w_skill, "availability": w_avail, "experience": w_exp, "rating": w_rating, "cost": w_cost, "certifications": w_certs}
    st.markdown("---")
    top_n = st.number_input("Top N candidates", value=5, min_value=1, max_value=20)

# Uploaded CSV 
if not uploaded:
    st.error("Please upload your resources CSV ")
    st.stop()

# Load uploaded CSV
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Error reading uploaded CSV: {e}")
    st.stop()

df = preprocess_df(df)



# Compose docs
docs_texts = [compose_doc_text(row) for _, row in df.iterrows()]
docs_meta = [row.to_dict() for _, row in df.iterrows()]

with st.spinner("Building embeddings & index (model download on first run)..."):
    emb_index, embeddings = build_index(docs_texts)
    bm25 = build_bm25(docs_texts)

if st.button("Run Matching"):
    t0 = time.time()
    st.info("Running hybrid retrieval, rerank and scoring...")

    query_info = extract_from_jd(jd_text)
    query_terms = " ".join(query_info.get("required_skills")) or jd_text

    candidate_indices = hybrid_search(query_terms, docs_texts, emb_index, embeddings, bm25=bm25, top_k_vector=50, top_k_bm25=30)
    st.write(f"Hybrid retrieved {len(candidate_indices)} candidates")

    if len(candidate_indices) == 0:
        st.warning("No candidates retrieved. Try relaxing JD or filters.")
        st.stop()

    top_indices, _ = rerank_topk(query_terms, docs_texts, candidate_indices, top_k=min(50,len(candidate_indices)))
    st.write(f"Reranked top {len(top_indices)} candidates")

    candidates_meta = []
    for idx in top_indices:
        meta = docs_meta[idx]
        feats = compute_feature_scores(query_info, meta)
        candidates_meta.append((idx, meta, feats))

    if not candidates_meta:
        st.warning("No candidates passed hard filters. Loosen filters and try again.")
        st.stop()

    costs = [m[1].get("Standard Cost",0) for m in candidates_meta]
    days = [m[2]["days_in_pool"] for m in candidates_meta]
    min_cost, max_cost = (min(costs), max(costs)) if costs else (0,1)
    min_days, max_days = (min(days), max(days)) if days else (0,365)

    final_list = []
    for idx, meta, feats in candidates_meta:
        feats["cost"] = feats.get("cost", meta.get("Standard Cost",0))
        score = compute_weighted_score(feats, weights, days_min=min_days, days_max=max_days, cost_min=min_cost, cost_max=max_cost)
        final_list.append((idx, meta, feats, score))

    final_sorted = sorted(final_list, key=lambda x: -x[3])[:top_n]

    rows = []
    for rank, (idx, meta, feats, score) in enumerate(final_sorted, start=1):
        evidence = {"skills": meta.get("Detailed Skill Set",""), "rating": meta.get("Resource Rating",""), "cost": meta.get("Standard Cost",""), "days_in_pool": meta.get("Days in Pool","")}
        gemini_summary = synthesize_with_gemini(meta, score, evidence) if get_gemini_key() else None
        summary = gemini_summary or f"Matches skills: {evidence.get('skills','')}. Experience: {meta.get('Total Experience (Years)','')} yrs."
        rows.append({
            "Rank": rank,
            "Resource ID": meta.get("Resource ID"),
            "Resource Name": meta.get("Resource Name"),
            "Primary Skill": meta.get("Primary Skill"),
            "Detailed Skills": meta.get("Detailed Skill Set"),
            "Score": round(score,4),
            "Days in Pool": meta.get("Days in Pool"),
            "Standard Cost": meta.get("Standard Cost"),
            "Resource Rating": meta.get("Resource Rating"),
            "Summary": summary
        })

    result_df = pd.DataFrame(rows)
    st.success(f"Top {len(result_df)} candidates (computed in {time.time()-t0:.1f}s)")
    st.table(result_df[["Rank","Resource ID","Resource Name","Primary Skill","Score","Days in Pool","Standard Cost","Resource Rating"]])

    with st.expander("Show full candidate summaries & evidence"):
        for _, r in result_df.iterrows():
            st.markdown(f"**{r['Rank']}. {r['Resource Name']} ({r['Resource ID']})** — Score: {r['Score']}")
            st.text(r["Summary"])
            st.write("Detailed Skills:", r["Detailed Skills"])
            st.write("---")

    csv_bytes = result_df.to_csv(index=False).encode()
    st.download_button("Download recommendations CSV", data=csv_bytes, file_name="recommended_candidates.csv", mime="text/csv")
