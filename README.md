# shk-turbo-mini

## Audience, UI, and Data

Target audience
- Primary: ML engineers and researchers building compact LLMs; developer teams running code-assist pipelines.
- Secondary: Applied NLP engineers, data scientists, open-source contributors, and SREs running reproducible training/CI.

User interface guidance
- CLI-first design: provide a single orchestration script with modular subcommands (`build`, `train`, `finetune`, `eval`) and a `--quick` flag for smoke runs.
- Optional web dashboard: developer-friendly dark theme, left nav (projects/runs/models), central logs/metrics, right-side artifacts panel.
- Agent UX: structured Plan / Act / Verify blocks, preview diffs, human approval required for write/apply actions.

Data & integrations
- Public sources: curated code corpora, Common Crawl subsets, Wikipedia, instruction-tuning datasets.
- Internal sources: private docs, code repos, and KBs ingested only with redaction and provenance metadata.
- Retrieval & storage: vector store (FAISS/Milvus) for embeddings, search index (Elasticsearch) for document retrieval.
- APIs: GitHub for repo snapshots and PR automation; CI runners for test-driven evaluation; telemetry via Prometheus/Grafana.
- Governance: PII detection/redaction, license checks for code, encryption at rest, access control, and full provenance tracking.