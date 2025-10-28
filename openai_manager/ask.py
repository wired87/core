from openai_manager import OAIC

def ask_chat(prompt, base64_string=None):
    print("====== AKS CHAT ======")
    payload = [
        {
            "type": "input_text",
            "text": prompt,
        },
    ]
    if base64_string is not None:
        payload.append(
            {
                "type": "input_file",
                #"filename": "draconomicon.pdf",
                "file_data": f"data:application/pdf;base64,{base64_string}",
            }
        )

    response = OAIC.responses.create(
        model="gpt-5",
        input=[{
            "role": "user",
            "content": payload,
        },
    ])

    response = response.output_text
    print(response)
    return response
