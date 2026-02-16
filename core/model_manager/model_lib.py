from core.qbrain_manager import get_qbrain_table_manager
from core.handler_utils import require_param, get_val
from gem_core.gem import Gem

class ModelManager:
    """
    Manager for handling AI Models within the system.
    """
    def __init__(self, qb, gem):
        self.gem = gem
        self.qb=qb
        self.table = "models"
        print("ModelManager initialized")

    def get_model(self, env_ids: list or str, user_id: str) -> str:
        try:
            model = self.qb.row_from_id(
                nid=env_ids,
                table=self.table,
                select="*",
                user_id=user_id
            )
            return model
        except Exception as e:
            print("Error getting model", e)


    def query(self, env_id: str, user_id: str, question: str) -> str:
        """
        Retrieve env entry -> select "model" -> usage static prompt for gem call -> return gem response.
        """
        print(f"ModelManager: Querying model for env {env_id}")
        
        # 1. Retrieve Env Entry
        # env_manager.retrieve_env_from_id returns {"envs": [entry]}
        from core.managers_context import get_env_manager
        env_data = get_env_manager().retrieve_env_from_id(env_id)
        if not env_data or "envs" not in env_data or not env_data["envs"]:
            return "Error: Environment not found."
            
        env_entry = env_data["envs"][0]
        
        model_content = env_entry.get("model")
        if not model_content:
             # Fallback/Debug
             model_content = env_entry.get("pattern", "No model or pattern found.")

        # 3. Static Prompt
        prompt = f"""
        You are an AI assistant analyzing a simulation model.
        
        MODEL STRUCTURE:
        {model_content}
        
        USER QUESTION:
        {question}
        
        Please answer the user's question based on the provided model structure.
        """
        
        # 4. Gem Call
        try:
            response = self.gem.ask(prompt)
            return response
        except Exception as e:
            print(f"Error querying Gemini: {e}")
            return f"Error processing query: {e}"

# Default instance for standalone use (no orchestrator context)
_default_model_manager = ModelManager(qb=get_qbrain_table_manager(), gem=Gem())
model_manager = _default_model_manager  # backward compat

# ==========================================
# HANDLERS
# ==========================================

def handle_query_model(data=None, auth=None):
    """Query the model for an environment with a natural language question. Required: user_id, env_id, question (auth or data)."""
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    env_id = get_val(data, auth, "env_id")
    question = get_val(data, auth, "question")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param(env_id, "env_id"):
        return err
    if err := require_param(question, "question"):
        return err
    from core.managers_context import get_model_manager
    answer = get_model_manager().query(env_id, user_id, question)
    return {"type": "QUERY_MODEL_RESPONSE", "data": {"answer": answer, "env_id": env_id}}
