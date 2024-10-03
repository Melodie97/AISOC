"""Build a simple LLM Application """

import os
import groq
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
groq_client = groq.Groq(api_key=GROQ_API_KEY)

system_prompt = "You are a helpful Virtual Assistant. \
                 Your goal is to provide useful and relevant \
                 responses to my request"

models=[
    "llama-3.1-405b-reasoning",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]

def generate(model, query, temperature=0):
    # Temperature is indication of how probablistic you want your response to be. 
    # If you want your response to be consistent, temp should be near 0 but if you want more
    # Creative room for the model it can be close to 1 up to 3. 
    # Temp near 1 is at higher risk of hallucination
    response = groq_client.chat.completions.create(
        model = model,
        # The messages argument usually takes in a list of dictionaries containing the:
        # System component, user component, and assistant component. The Assistant component will be
        # Important when we start looking at persisting conversational history.
        messages = [
            {'role':'system', 'content':system_prompt},
            {'role':'user', 'content':query},
        ],
        response_format = {'type':'text'}, #default is text, can be changes to json 
        temperature = temperature
    )

    answer = response.choices[0].message.content

    return answer


# if __name__ == "__main__":
#     model = models[1]
#     query = "which is bigger? 9.11 or 9.9?"
#     response = generate(model, query, temperature=1)
#     print(response)
