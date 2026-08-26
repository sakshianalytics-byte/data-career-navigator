# Data Career Navigator — Your Career in the Age of AI

An open-source, evidence-based career-transition agent for data & AI professionals.

It answers two questions:

1. **Career Navigator** — *"Given my career history and where the AI job market is heading, what should my next role be?"*
2. **JD Analyzer** — *"For this company's job description, how much can AI take over, which tasks does AI handle, and which stay human?"*

The core idea: career advice is treated as a **data problem**, not a black-box chatbot answer.
Every role is a **skill vector**, and every score (transition fit, AI exposure, remote fit) is
computed with transparent math you can inspect. The product is framed around **career
evolution, not job replacement**.

---

## Why this is different

- **Explainable by design.** Every recommendation shows *"Why am I being recommended this role?"* with the exact numbers behind it.
- **Runs for free, fully offline.** The scoring engine is deterministic Python — no paid API key required.
- **Optional LLM layer.** Plug in your own OpenAI / Anthropic / local model for richer text. Never mandatory.
- **Open data only.** Seed dataset is bundled and MIT-friendly; documented path to swap in full [O*NET](https://www.onetcenter.org/) and [BLS](https://www.bls.gov/) data.

---

## What it produces

**Career Navigator**
1. Where you are today (skill profile: strong / moderate / emerging)
2. Which parts of your role AI will change (task-level AI impact table)
3. Top realistic next roles, each with a **Transition Score** and **Future Fit Score**
4. Skill gap for each role
5. Remote-work fit
6. A 90-day transition plan (3 x 30-day phases)
7. Recommended portfolio projects
8. Full evidence breakdown for every score

**JD Analyzer**
1. Overall **AI takeover %** for the role
2. Actions AI can take over
3. Actions that remain human-critical
4. Human + AI hybrid tasks
5. Task-level breakdown table with rationale

---

## Tech stack (all free / open-source)

| Layer | Tool | Purpose |
|-------|------|---------|
| Language | Python | Core logic |
| Notebook | Jupyter | Shareable, runs on Colab/Binder |
| Math | NumPy | Skill-vector distances & similarity |
| Tables | Pandas | Task-impact breakdowns |
| Data | JSON | Bundled open seed dataset |
| Web (optional) | Streamlit | Free-to-host demo app |
| LLM (optional) | OpenAI / Anthropic / Ollama | Richer narrative text |

---

## Project structure

```
AI-landscape-job-market/
├── career_navigator.ipynb    # Main notebook — start here
├── app.py                    # Optional Streamlit web app
├── src/
│   ├── data_loader.py        # Loads the seed dataset
│   ├── engine.py             # Deterministic scoring engine
│   └── jd_analyzer.py        # Job-description AI-exposure analyzer
├── data/
│   ├── roles.json            # Role skill vectors + demand/growth/remote
│   ├── skills.json           # Canonical skill list + categories
│   └── ai_task_impact.json   # Task -> AI-impact ratings + rationale
├── requirements.txt
├── LICENSE                   # MIT
└── README.md
```

---

## Quick start

### Google Colab (easiest — upload one file)

The notebook is **self-contained**. Just upload `career_navigator.ipynb` to
[Google Colab](https://colab.research.google.com/) and run all cells. The first
cell installs `numpy`/`pandas` and writes the engine + data into the runtime
automatically — you do **not** need to upload the `src/` or `data/` folders.

### Local

```bash
pip install -r requirements.txt

# Option A: open the notebook
jupyter notebook career_navigator.ipynb

# Option B: run the web app
streamlit run app.py
```

Host the app for free on **Streamlit Community Cloud** (point it at your GitHub fork).

> The notebook is generated from the `src/` modules and `data/` files by
> `build_notebook.py`. If you change the engine or data, run
> `python build_notebook.py` to refresh the self-contained notebook.

---

## Data & methodology note

The bundled dataset is a **curated seed** encoding public, well-known concepts about data/AI roles,
their skills, and AI task exposure. Scores are illustrative and computed transparently. To make the
engine authoritative, swap in the full public datasets:

- **O*NET** — occupation tasks, skills, transferable skills, related occupations
- **BLS Employment Projections** — occupation growth / demand
- An **open-licensed job-postings dataset** (e.g. a public Kaggle dataset) — market demand & remote share

See `src/data_loader.py` for the schema to match when replacing the seed data.

## License

MIT — see [LICENSE](LICENSE).
