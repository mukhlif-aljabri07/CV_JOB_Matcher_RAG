# CV ↔ Job Match Analyzer

A Retrieval-Augmented Generation (RAG) tool that compares a candidate's CV against a job description and produces an honest assessment of fit — strengths, gaps, and a direct verdict — grounded in the actual text of both documents rather than generic advice.

Built with Python, Sentence-Transformers, ChromaDB, and Groq (Llama 3.3 70B).

---

## Why I built this

I built this to face a question every job seeker quietly dreads: *"Am I actually a fit for this, or am I about to waste my time?"* I'm in that position right now, and I know I'm not alone in it — so instead of guessing, I built a tool that reads my CV against a real job description and tells me honestly where I match, where my gaps are, and whether it's worth applying. It turns an anxious guess into something I can actually look at and act on.

---

## What it does

- Upload a CV (PDF) and paste a job description.
- The tool retrieves the most relevant parts of each document and asks an LLM to compare them.
- Returns a structured analysis: **Match** (what you genuinely have), **Gaps** (what you're missing), and an **Honest Verdict** (strong fit / stretch / mismatch).

The analysis is grounded only in the text of the two documents — the model is instructed to reason from the retrieved evidence, not from general assumptions.

---

## How it works (architecture)

The pipeline has five stages:

1. **Extraction** — CV text is extracted from the uploaded PDF; the job description is taken as raw text.
2. **Chunking** — each document is split into ~150-word chunks so retrieval stays precise.
3. **Embedding** — each chunk is converted into a vector using `all-MiniLM-L6-v2` (a 384-dimension sentence-embedding model).
4. **Storage & retrieval** — chunks and embeddings are stored in ChromaDB, **tagged by source** (`cv` or `job`). At query time, the system retrieves relevant chunks from each source *separately* using a metadata filter.
5. **Reasoning** — the labeled CV and job chunks are passed to an LLM (Llama 3.3 70B via Groq), which compares them and produces the assessment.

### The key design decision

Working with two documents isn't the same as working with one. The key design decision is that each chunk is tagged by its source (`cv` or `job`) when it's stored, so retrieval can pull relevant chunks from each document *separately* rather than mixing them. This matters because the whole point of the tool is to compare "what the candidate has" against "what the job wants" — if I merged the two documents into one representation, I'd destroy exactly that distinction. So retrieval's job is only to gather labeled evidence from each side; the actual comparison and reasoning is left to the LLM. That separation — retrieval organizes, the model reasons — is what makes the assessment possible.

---

## Known limitations

- **Semantic search struggles with exact factoid lookups.** Pure embedding-based retrieval matches *meaning*, not keywords, so questions like "what is the project title?" can underperform because the question phrasing isn't semantically similar to the answer text. A production version would add hybrid (keyword + semantic) search.
- **Short CVs produce few chunks**, which limits retrieval granularity on the candidate side.
- **LLM output varies** slightly between runs on borderline cases (e.g. "stretch" vs "mismatch"), as is expected with generative models.

---

## Tech stack

| Component | Tool |
|-----------|------|
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent, local) |
| LLM | Llama 3.3 70B via Groq API |
| Interface | Gradio |
| PDF extraction | pypdf |

---

## Running it locally

```bash
pip install -r requirements.txt
```

Set your Groq API key (get one free at console.groq.com):

```bash
# in a .env file
GROQ_API_KEY=your_key_here
```

Then run:

```bash
python app.py
```

The app launches at `http://127.0.0.1:7860`.

---

## What I'd build next

- **Rank multiple job postings at once** — the real problem isn't matching one job, it's deciding which of many are worth applying to. Ranking a batch by fit would turn this into a job-hunt triage tool.
- **Accept the job description as a PDF upload**, not just pasted text.
- **Add hybrid (keyword + semantic) search** to fix the factoid-retrieval limitation noted above — so exact terms like specific tools or titles aren't missed.
