"""
Agents package for LitSynth AI multi-agent pipeline.

Design Patterns:
  - Pattern 1 (Orchestrator-Worker): OrchestratorAgent decomposes the task;
    RouterAgent acts as a worker to classify intent.
  - Pattern 2 (ReAct/Tool-Use): RetrieverAgent uses a tool (FAISS search)
    in a Thought → Action → Observation loop.
  - Pattern 3 (Reflection/Self-Critique): CriticAgent reviews SynthesizerAgent's
    output and scores factual grounding.

Agent-to-Agent (A2A) Communication:
  All agents append structured JSON messages to st.session_state["a2a_trace"].
"""
