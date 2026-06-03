# AI-Powered SDLC Pipeline

An intelligent multi-agent system that automates the full software development lifecycle — from a natural language feature request to production deployment — using LangGraph, Google Gemini, and Jira.

## What It Does

You describe a feature or bug in plain English. Six AI agents handle the rest:

```
You (text input)
  → Agent 1 (PM)       — writes user story → creates Jira ticket
  → Agent 2 (Dev)      — reads ticket → RAG over codebase → writes code → pushes to GitHub
  → Agent 5 (Test)     — runs pytest → reviews code quality → pass/fail report
  → Agent 3 (Build)    — writes Dockerfile → containerizes (only if tests pass)
  → Agent 4 (Deploy)   — deploys to Minikube
  → Agent 6 (Prod)     — tags release → deploys to production namespace
```

Every step updates the Jira ticket with a full audit trail: who, what, why, when, where.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Google Gemini 2.5 Flash (free tier) |
| Ticket management | Jira REST API |
| Code retrieval | RAG with vector embeddings |
| Version control | GitHub API (PyGithub) |
| Containerization | Docker |
| Local deployment | Minikube (Kubernetes) |
| CI/CD | GitHub Actions |
| Language | Python 3.13 |

---

## Project Structure

```
sdlc-ai-pipeline/
├── pipeline.py              # Entry point — wires all agents into LangGraph graph
├── agents/
│   ├── agent1_pm.py         # PM Agent — Jira ticket creation
│   └── agent2_dev.py        # Dev Agent — RAG + code changes (in progress)
├── tools/
│   └── jira_tools.py        # Jira API functions (create, comment, update status)
├── state/
│   └── pipeline_state.py    # Shared StateSDLC TypedDict
├── config/
│   └── settings.py          # Loads environment variables
├── sample_app/              # Target app Agent 2 practices on (separate repo)
├── practice/                # LangGraph practice exercises
├── requirements.txt
└── .env                     # Never committed — see Environment Variables below
```

---

## Setup

### Prerequisites
- Python 3.13+
- A Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))
- A Jira free account ([atlassian.com/software/jira/free](https://www.atlassian.com/software/jira/free))

### Installation

```bash
# Clone the repo
git clone https://github.com/sakshitokekar/sdlc-ai-pipeline.git
cd sdlc-ai-pipeline

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
python3 -m pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key
JIRA_EMAIL=your_atlassian_email
JIRA_API_TOKEN=your_jira_api_token
JIRA_BASE_URL=https://your-domain.atlassian.net
```

### Run the Pipeline

```bash
python3 pipeline.py
```

---

## Current Status

| Agent | Status | Description |
|---|---|---|
| Agent 1 — PM | ✅ Complete | Creates real Jira tickets via REST API |
| Agent 2 — Dev | 🔄 In Progress | RAG + code changes + GitHub push |
| Agent 3 — Build | ⏳ Planned | Dockerfile + containerization |
| Agent 4 — Deploy | ⏳ Planned | Minikube deployment |
| Agent 5 — Test | ⏳ Planned | pytest + code review |
| Agent 6 — Prod | ⏳ Planned | Production deployment |

---

## Known Limitations

- **Personal Jira account** — using personal Atlassian account instead of a service account. In production, a dedicated service account with minimum permissions should be used.
- **No human-in-the-loop yet** — agents run fully autonomously. Human review checkpoints (after Agent 1 and Agent 2) are planned for Phase 4.
- **No conversation loop in Agent 1** — Agent 1 makes assumptions for missing fields rather than asking clarifying questions.
- **MemorySaver checkpointer** — state is stored in memory only. Will be upgraded to SqliteSaver/PostgresSaver for production persistence.

---

## Future Work

- Human-in-the-loop review checkpoints using LangGraph `interrupt()` after Agent 1 (ticket approval) and Agent 2 (code review)
- Conversation loop in Agent 1 for clarifying vague requirements
- Input/Output/Private state separation for clean user-facing API
- Upgrade checkpointer: MemorySaver → SqliteSaver → PostgresSaver
- Streaming output using `stream_mode="updates"` for real-time agent progress
- Full audit trail: every Jira ticket updated with who/what/why/when/where per agent

---

## Branching Strategy

```
main          ← stable, production-ready releases only
└── develop   ← integration branch, all features merge here first
    ├── feature/phase1-pm-agent
    ├── feature/phase2-dev-agent
    └── feature/phase3-build-agent  (coming soon)
```

---

## Learning Context

This project was built as a hands-on learning project to develop applied AI engineering skills. Each agent was built incrementally — study the concept, then build it.

Built with: Python · LangGraph · Google Gemini API · Jira REST API · GitHub API · Docker · Minikube
