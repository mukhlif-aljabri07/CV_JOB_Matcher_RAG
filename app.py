import os
import re
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq


import gradio_client.utils as _gc_utils

_orig_get_type = _gc_utils.get_type
def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)
_gc_utils.get_type = _safe_get_type

_orig_json_schema = _gc_utils._json_schema_to_python_type
def _safe_json_schema(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_json_schema(schema, defs)
_gc_utils._json_schema_to_python_type = _safe_json_schema
# --- end patch ---


# Setup 
model = SentenceTransformer("all-MiniLM-L6-v2")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chroma_client = chromadb.Client()



# Core functions

def extract_pdf_text(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def chunk_text_from_string(text, chunk_size=150):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks


def embed_chunks(chunks):
    return [model.encode(chunk) for chunk in chunks]


def add_document(text, source_label, collection):
    text = re.sub(r"\s+", " ", text)
    doc_chunks = chunk_text_from_string(text, chunk_size=150)
    doc_embeddings = [emb.tolist() for emb in embed_chunks(doc_chunks)]
    ids = [f"{source_label}_chunk_{i}" for i in range(len(doc_chunks))]
    metadatas = [{"source": source_label} for _ in range(len(doc_chunks))]
    collection.add(
        documents=doc_chunks,
        embeddings=doc_embeddings,
        ids=ids,
        metadatas=metadatas,
    )


def retrieve_for_matching(query, collection, top_k=3):
    query_embedding = model.encode(query).tolist()
    cv_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"source": "cv"},
    )
    job_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"source": "job"},
    )
    return cv_results["documents"][0], job_results["documents"][0]


def analyze_match(cv_chunks, job_chunks):
    cv_context = "\n".join(cv_chunks)
    job_context = "\n".join(job_chunks)

    prompt = f"""You are a career advisor. Compare the candidate's CV against the job requirements below and give an honest assessment.
CANDIDATE'S CV (relevant parts):
{cv_context}
JOB REQUIREMENTS (relevant parts):
{job_context}
Provide:
1. MATCH: What does the candidate genuinely have that this role needs?
2. GAPS: What key requirements does the candidate NOT appear to meet?
3. HONEST VERDICT: Is this a strong fit, a stretch, or a mismatch? Be direct, not encouraging for its own sake.
Base your answer ONLY on what's in the CV and job text above."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Pipeline + Gradio interface
# ---------------------------------------------------------------------------

def analyze_cv_vs_job(cv_file, job_text):
    if cv_file is None:
        return "Please upload a CV (PDF)."
    if not job_text or job_text.strip() == "":
        return "Please paste a job description."

    cv_text = extract_pdf_text(cv_file)

    try:
        chroma_client.delete_collection(name="cv_job_matcher")
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name="cv_job_matcher")

    add_document(cv_text, "cv", collection)
    add_document(job_text, "job", collection)

    query = "what are the key skills and requirements for this role?"
    cv_chunks, job_chunks = retrieve_for_matching(query, collection)
    return analyze_match(cv_chunks, job_chunks)


demo = gr.Interface(
    fn=analyze_cv_vs_job,
    inputs=[
        gr.File(label="Upload your CV (PDF)", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Paste the job description", lines=12,
                   placeholder="Paste the full job posting here..."),
    ],
    outputs=gr.Markdown(label="Match Analysis"),
    title="CV ↔ Job Match Analyzer",
    description="Upload your CV and paste a job description. Get an honest assessment of your fit, your strengths, and your gaps — powered by RAG.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
