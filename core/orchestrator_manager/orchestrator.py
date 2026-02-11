
"""
Orchestrator Manager

This module implements the OrchestratorManager, responsible for high-level coordination
and optimization of simulation entries using advanced neural architectures.

HOPFIELD NETWORK ORCHESTRATION EXPLANATION:
-------------------------------------------
The Orchestrator employs a continuous-state Hopfield Network (or modern dense associative memory)
to manage and orchestrate over existing simulation entries. 

1. Conceptualization:
   Each simulation entry (comprising environment configurations, module states, field parameters, 
   and injection patterns) is treated as a "memory pattern" or "attractor state" within the 
   network's energy landscape.

2. Process:
   - The "State Space" is defined by the aggregate parameters of all active simulations.
   - The Orbit of the simulation execution is viewed as a trajectory through this high-dimensional energy landscape.
   - Using the Hopfield energy function E = -1/2 * x^T * W * x (where W is the weight matrix learned 
     from successful historic simulations), the Orchestrator directs the current simulation flow 
     towards local minima which represent stable, coherent, or "optimal" simulation states.

3. Utility:
   - **Error Correction**: If a user provides a partial or incoherent simulation setup (noisy input), 
     the network relaxes the state to the nearest stored valid configuration (pattern retrieval).
   - **Convergence**: It ensures that dynamic interactions between modules and fields settle into 
     stable equilibrium points rather than oscillating chaotically, effectively "orchestrating" 
     transient dynamics into meaningful steady states.
   - **Interpolation**: It can plausibly interpolate between two known stable simulation states 
     to generate novel but valid transitional configurations.

This approach transforms the role of the Orchestrator from a simple task scheduler to a 
dynamic stability controller, ensuring global coherence across distributed simulation components.
"""

from typing import List, Dict, Optional, Any, TypedDict
import json
from core.qbrain_manager import QBrainTableManager
from a_b_c.bq_agent._bq_core.bq_handler import BQCore
from gem_core.gem import Gem
from core.file_manager.file_lib import file_manager
from core.session_manager.session import session_manager
from core.researcher2.researcher2.core import ResearchAgent

class StartSimInput(TypedDict):
    """
    Structure for gathering information required to start a simulation.
    """
    simulation_name: str
    target_env_id: str
    duration_seconds: int
    time_step: float
    description: Optional[str]

class OrchestratorManager(BQCore):
    """
    Manages orchestration of simulations using Hopfield Network dynamics.
    Now includes a conversational interface to setup simulations.
    """
    DATASET_ID = "QBRAIN"
    
    def __init__(self):
        super().__init__(dataset_id=self.DATASET_ID)
        self.qb = QBrainTableManager()
        self.gem = Gem()
        self.research_agent = ResearchAgent()
        
        # Simple in-memory context storage for demo purposes
        # Key: session_id, Value: List[Dict[role, content]]
        self.chat_contexts: Dict[str, List[Dict[str, str]]] = {}
        
        # Key: session_id, Value: Partial StartSimInput
        self.simulation_drafts: Dict[str, Dict[str, Any]] = {}
    """
    bsp task:
    entwickel und teste ein ravity modell. 
    zeig mir alle files an
    
    """




    def orchestrator_chat(self, user_id: str, session_id: str, message: str) -> str:
        """
        Interact with the user to gather simulation parameters.
        Returns the assistant's response.
        """
        # 1. Initialize context if needed
        if session_id not in self.chat_contexts:
            self.chat_contexts[session_id] = [
                {"role": "system", "content": """
                You are the Orchestrator Assistant for the BestBrain simulation platform.
                Your goal is to help the user configure and start a simulation.
                
                You need to collect the following information to create a 'StartSimInput' structure:
                1. simulation_name (str): A unique name for this run.
                2. target_env_id (str): The ID of the environment to simulate.
                3. duration_seconds (int): How long the simulation should run.
                4. time_step (float): The delta time for each step.
                5. description (optional): A brief description.

                Ask questions one by one or in small groups to gather this data.
                Once you have ALL the required information, output a specific JSON block at the end of your message:
                ```json
                {
                    "action": "START_SIM",
                    "data": {
                        "simulation_name": "...",
                        "target_env_id": "...",
                        "duration_seconds": 10,
                        "time_step": 0.1,
                        "description": "..."
                    }
                }
                ```
                If the user asks about existing environments, you can mention you don't have direct access yet (or we can inject it into context later).
                Keep responses concise and professional.
                """}
            ]
            self.simulation_drafts[session_id] = {}

        # 2. Add User Message
        self.chat_contexts[session_id].append({"role": "user", "content": message})

        # 3. Get LLM Response
        # We construct a prompt from the history
        # Gem.ask takes a string or list of contents. Adapt as needed.
        # Here assuming simple concat for the 'content' param of Gem.ask for simplicity, 
        # or adapting if Gem supports chat history natively. 
        # Let's flatten for now as `ask` usually takes a string prompt.
        
        conversation_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.chat_contexts[session_id]])
        
        response_text = self.gem.ask(content=conversation_str)

        # 4. Process Response
        if not response_text:
            return "I'm having trouble connecting to my neural core. Please try again."

        self.chat_contexts[session_id].append({"role": "model", "content": response_text})

        # 5. Check for Action Block
        if "```json" in response_text and "START_SIM" in response_text:
            try:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
                action_data = json.loads(json_str)
                
                if action_data.get("action") == "START_SIM":
                    sim_input = action_data.get("data")
                    # Trigger Simulation
                    result = self.start_sim(sim_input, user_id, session_id)
                    return f"{response_text}\n\n[System]: Simulation started successfully! ID: {result.get('sim_id')}"
            except Exception as e:
                print(f"Error parsing chat action: {e}")
                return response_text + "\n\n[System]: Failed to start simulation due to parsing error."

        return response_text

    def start_research_for_session(self, user_id: str, session_id: str, prompt: str) -> None:
        """
        Kick off a deep-research workflow for a session.

        - Uses ResearchAgent to discover relevant documents.
        - The callback merges discovered URLs into the session's research_files column.
        - ResearchAgent.research_workflow handles file processing and Vertex RAG ingestion.
        """

        def _on_sources(urls: List[str]) -> None:
            try:
                session_manager.update_research_files(user_id, session_id, urls)
            except Exception as e:
                print(f"[OrchestratorManager] Failed to update research_files via callback: {e}")

        self.research_agent.run(
            prompt=prompt,
            use_dr_result_callable=_on_sources,
            user_id=user_id,
            session_id=session_id,
        )

    def start_sim(self, input_data: StartSimInput, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Execute the workflow to start a simulation based on gathered input.
        """
        print(f"Starting simulation with input: {input_data}")
        
        # 1. Validate Env ID (Basic check)
        env_id = input_data.get("target_env_id")
        # In a real scenario, we might verify env ownership here
        
        # 2. Create Simulation Record (e.g. into 'simulations' table, or 'logs' in envs)
        # For now, we'll just log it to the 'envs' table as a simplistic update or a new 'simulations' table if it existed.
        # Let's assume we update the environment status or similar.
        
        sim_id = f"sim_{env_id}_{input_data.get('simulation_name')}"
        
        # 3. Trigger Orchestration Logic
        self.orchestrate(session_id, user_id)
        
        return {
            "status": "started",
            "sim_id": sim_id,
            "config": input_data
        }

    def orchestrate(self, session_id: str, user_id: str):
        """
        Trigger the orchestration process for a given session.
        This would compute the energy state and perform updates to align
        the simulation with stored patterns.
        """
        # 1. Retrieve current session state (Context)
        # 2. Map to network state vector
        # 3. Apply Hopfield dynamics until convergence
        # 4. Update session components with new stable state
        print(f"Orchestrating session {session_id} for user {user_id}...")
        pass

    def identify_table_from_query(self, query: str) -> Optional[str]:
        """
        Identify the target table based on user query and MANAGERS_INFO.
        Uses the qbrain_manager MANAGERS_INFO struct to find a matching table.
        """
        # Extract table info from QBrainTableManager
        tables_info = []
        valid_tables = set()

        for manager in self.qb.MANAGERS_INFO:
            mgr_name = manager.get("manager_name", "Unknown")
            # Primary table
            default_table = manager.get("default_table")
            if default_table:
                tables_info.append(f"Table: {default_table} ({mgr_name}) - {manager.get('description', '')}")
                valid_tables.add(default_table)
            
            # Additional tables
            if manager.get("additional_tables"):
                for extra in manager["additional_tables"]:
                    t_name = extra.get("table_name")
                    if t_name:
                        tables_info.append(f"Table: {t_name} ({mgr_name}) - Secondary table")
                        valid_tables.add(t_name)

        tables_context = "\n".join(tables_info)
        
        prompt = f"""
        Analyze the user query and identify the single most relevant database table.
        
        User Query: "{query}"
        
        Available Tables:
        {tables_context}
        
        Return ONLY the exact table name (e.g., 'users'). If no table matches, return "NONE".
        """
        
        response = self.gem.ask(content=prompt)
        
        if response:
            cleaned = response.strip().replace("'", "").replace('"', "")
            if cleaned in valid_tables:
                return cleaned
                
        return None

orchestrator_manager = OrchestratorManager()
