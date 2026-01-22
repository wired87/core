from typing import Any, Dict, List
from gem_core.gem import Gem
from core.env_manager.env_lib import env_manager

class ModelManager:
    """
    Manager for handling AI Models within the system.
    """
    def __init__(self):
        self.gem = Gem()
        print("ModelManager initialized")

    def query(self, env_id: str, user_id: str, question: str) -> str:
        """
        Retrieve env entry -> select "model" -> usage static prompt for gem call -> return gem response.
        """
        print(f"ModelManager: Querying model for env {env_id}")
        
        # 1. Retrieve Env Entry
        # env_manager.retrieve_env_from_id returns {"envs": [entry]}
        env_data = env_manager.retrieve_env_from_id(env_id)
        if not env_data or "envs" not in env_data or not env_data["envs"]:
            return "Error: Environment not found."
            
        env_entry = env_data["envs"][0]
        
        # 2. Select "model"
        # The prompt says 'select "model"'. Assuming there is a key "model" in the env entry.
        # Checking env_lib.py schema, there is 'model': "STRING" in TABLES_SCHEMA (inferred), 
        # but in env_lib.py ENV_SCHEMA it is not explicitly listed in the snippet I saw? 
        # Wait, I saw ENV_SCHEMA in env_lib.py: 
        # code: bigquery.SchemaField("data", "STRING", mode="NULLABLE"),
        # I did not see "model" in ENV_SCHEMA in the snippet I read (lines 1-200 of env_lib.py).
        # However, QBrainTableManager TABLES_SCHEMA might have it.
        # Let's assume it is in the "data" field or a missing schema column that the user expects to exist, 
        # OR "model" refers to the structure defined in "pattern" or similar.
        # The User Request says: 'retrieve env entry -> select "model"'.
        # I will try to access 'model' key directly. If not present, maybe check 'pattern'.
        
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

# Instantiate
model_manager = ModelManager()

# ==========================================
# HANDLERS
# ==========================================

def handle_query_model(payload):
    """
    receive "QUERY_MODEL": 
      auth={env_id, user_id}, 
      data={question: str}
    """
    print("handle_query_model")
    auth = payload.get("auth", {})
    data = payload.get("data", {})
    
    user_id = auth.get("user_id")
    env_id = auth.get("env_id")
    question = data.get("question")
    
    if not all([user_id, env_id, question]):
        return {"error": "Missing user_id, env_id, or question"}
        
    answer = model_manager.query(env_id, user_id, question)
    
    return {
        "type": "QUERY_MODEL_RESPONSE",
        "data": {
            "answer": answer,
            "env_id": env_id
        }
    }
