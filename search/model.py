import google.generativeai as genai

class Model():
    def __init__(self, 
                 api_key : str
                 ):
        genai.configure(api_key=api_key)
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=520,
                temperature=1.0)
        self.model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config, safety_settings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE",
                        },
                    ])
    def answer(self, 
               query : str, 
               context : str):
        messages = [
        {'role':'user',
         'parts': [f"""
         Answer the following query using the provided context. If you cannot answer then simply return ```Here are the documents related to {query}.``` modify the query in the failure output accordingly.
                 
                 Query:
             {query}
            Context:
            {context}
             """]}
            ]
        for response in self.model.generate_content(messages, stream=True):
            yield response.text