from qbrain.chat_manger.prompts.explanation_bestbrain import EXPLANATION


def get_welcome_prompt(
        platform_name="botworld.cloud",
        feature_list=[
            "create_config",
            "ask",
            "run_simulation",
            ""
        ]
):
    return f"""
    You are a helpful assistant.  
    Your task is to generate a short, friendly, and professional welcome message for a new user on our platform.  
    The message must:  
    - Begin with a warm greeting.  
    - Mention the platform name: {platform_name}.  
    - Explain sense of the platform in one sentence following an explanation: {EXPLANATION} 
    - Clearly list all available features from the following list: {feature_list}.  
    - Use simple, positive, and motivating language.  
    - End with an encouraging note, such as “Enjoy exploring!” or “We’re excited to have you here!”.  
    
    Parameters:  
    - platform_name: The name of the platform.  
    - feature_list: A comma-separated list of all features the user can access.  

    """



