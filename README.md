# ⚡ HR Leave & Expense Policy Adjudicator

An autonomous HR compliance assistant powered by a **3-Agent LangGraph Pipeline**, **ChromaDB Vector RAG**, and **OpenAI GPT-OSS models (`120b` & `20b`)** hosted on Groq LPU inference.

## 🏗️ Architecture
[Employee Submission / Claim Request]
│
▼
┌───────────────────────────────────┐
│     Agent 1: Intake Specialist    │ ➔ Normalizes amounts, dates, and receipts
└─────────────────┬─────────────────┘
▼
┌───────────────────────────────────┐
│     Agent 2: Compliance Auditor   │ ➔ Queries ChromaDB RAG & audits policy caps
└─────────────────┬─────────────────┘
▼
┌───────────────────────────────────┐
│     Agent 3: Resolution Officer   │ ➔ Generates financial table & employee email
└─────────────────┬─────────────────┘
▼
[Streamlit UI Display]
