import os
import streamlit as st
import chromadb
from typing import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

st.set_page_config(
    page_title="HR Policy Adjudicator",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ HR Leave & Expense Adjudicator")
st.markdown("Automated compliance system orchestrated with **LangGraph**, **Groq (OpenAI GPT-OSS)**, and **ChromaDB RAG**.")

# 1. Fetch Credentials Securely from Streamlit Secrets
if "GROQ_API_KEY" not in st.secrets:
    st.error(
        "Missing `GROQ_API_KEY`! Please configure it in `.streamlit/secrets.toml` locally, "
        "or in the Streamlit Cloud dashboard under Settings > Secrets."
    )
    st.stop()

groq_api_key = st.secrets["GROQ_API_KEY"]

# 2. Model Selection Sidebar
selected_model = st.sidebar.radio(
    "Select Model Engine:",
    options=["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    index=0,
    help="120B provides deeper policy reasoning; 20B provides ultra-high throughput."
)

# 3. ChromaDB Vector Store Query Helper
def query_hr_rag(search_query: str) -> str:
    if not os.path.exists("./chroma_db"):
        return "Policy database not found. Please run `python rag_setup.py` first."
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="hr_policies")
    results = collection.query(query_texts=[search_query], n_results=2)
    
    if not results or not results["documents"][0]:
        return "No specific policy clause found."
    return "\n---\n".join(results["documents"][0])

# 4. LangGraph State Definition (Standard TypedDict, No Pydantic schemas needed)
class AdjudicatorState(TypedDict):
    raw_submission: str
    parsed_items: str
    policy_context: str
    audit_findings: str
    final_decision: str

# 5. Initialize Groq Chat Model
llm = ChatGroq(
    model=selected_model,
    api_key=groq_api_key,
    temperature=0.1
)

# 6. Define the 3 Agents as Graph Nodes

def agent_intake_node(state: AdjudicatorState) -> dict:
    """Agent 1: Extracts and structures all expense line items and leave days."""
    prompt = f"""
    You are the HR Claims Intake Specialist.
    Extract every expense item and leave request from this submission into a clean, numbered list.
    For each item, specify: Description, Date, Amount, Expense Category, and whether an attachment/receipt exists.
    Do not verify policy compliance yet.

    Submission:
    {state['raw_submission']}
    """
    response = llm.invoke(prompt)
    return {"parsed_items": str(response.content)}


def agent_auditor_node(state: AdjudicatorState) -> dict:
    """Agent 2: Retrieves policy rules from ChromaDB and audits each line item."""
    retrieved_policies = query_hr_rag(state["parsed_items"])
    
    prompt = f"""
    You are the HR Policy Compliance Auditor.
    Audit these claim items against the retrieved company guidelines.

    Policy Guidelines:
    {retrieved_policies}

    Extracted Items:
    {state['parsed_items']}

    For each item, determine:
    1. Status: (Compliant / Exceeded Limit / Disallowed / Missing Proof)
    2. Citing Rule: Explicitly cite rule numbers and dollar caps from guidelines.
    3. Variance: Any disallowed or excess amount.
    """
    response = llm.invoke(prompt)
    return {
        "policy_context": retrieved_policies,
        "audit_findings": str(response.content)
    }


def agent_resolution_node(state: AdjudicatorState) -> dict:
    """Agent 3: Drafts the executive breakdown table and the employee response letter."""
    prompt = f"""
    You are the HR Communications Officer.
    Synthesize these audit findings into a formal adjudication response:

    Audit Findings:
    {state['audit_findings']}

    Produce:
    1. Executive Status: (Approved / Partially Approved / Needs Revision)
    2. Financial Breakdown Table: Columns for Item, Claimed, Approved, Disallowed, and Justification.
    3. Employee Letter: A clear, polite email explaining the decision and what documentation is required to release pending balances.
    """
    response = llm.invoke(prompt)
    return {"final_decision": str(response.content)}


# 7. Build the LangGraph StateGraph Pipeline
def build_adjudication_graph():
    workflow = StateGraph(AdjudicatorState)

    # Register Nodes
    workflow.add_node("intake_agent", agent_intake_node)
    workflow.add_node("auditor_agent", agent_auditor_node)
    workflow.add_node("resolution_agent", agent_resolution_node)

    # Connect Edges: START -> Intake -> Auditor -> Resolution -> END
    workflow.add_edge(START, "intake_agent")
    workflow.add_edge("intake_agent", "auditor_agent")
    workflow.add_edge("auditor_agent", "resolution_agent")
    workflow.add_edge("resolution_agent", END)

    return workflow.compile()


# 8. Main Form Interface
default_claim = """Employee: Rahul Verma (Sales Lead)
Submission Date: October 14, 2026
Expenses:
- Client Dinner on Sept 10: $95.00 total ($65 food + $30 wine). Itemized receipt attached.
- Solo Airport Cab on Sept 9: $52.00 via Uber Black. Receipt attached.
- Hotel Stay: $170.00/night for 2 nights. Invoice attached.
Leave Request:
- 4 days of Sick Leave (Oct 20-23). No doctor note attached."""

claim_input = st.text_area("Employee Claim / Leave Submission:", value=default_claim, height=180)

if st.button("Run Multi-Agent Adjudication", type="primary"):
    with st.spinner(f"Executing LangGraph pipeline with {selected_model}..."):
        try:
            app_graph = build_adjudication_graph()
            
            initial_state: AdjudicatorState = {
                "raw_submission": claim_input,
                "parsed_items": "",
                "policy_context": "",
                "audit_findings": "",
                "final_decision": ""
            }

            final_output = app_graph.invoke(initial_state)

            st.success("Adjudication Complete!")
            st.markdown(final_output["final_decision"])

            with st.expander("Inspect Intermediate Multi-Agent State"):
                st.subheader("Agent 1 (Intake Specialist) - Parsed Items:")
                st.write(final_output["parsed_items"])
                st.subheader("Agent 2 (Auditor) - RAG Context & Audit:")
                st.write(final_output["audit_findings"])

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
