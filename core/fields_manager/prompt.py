def extract_fields_prompt(
        instructions, params, fallback_params
):
    return f"""
    Your task is to create as much fields of the provided content and return them 
    in the specified input field format 
    capturing all aprams and other information.

    
    AVAILABLE PARAMS:
    {params}
    
    FALLBACK PARAMS:
    {fallback_params}
    
    USER INSTRUCTIONS:
    {instructions}
    """