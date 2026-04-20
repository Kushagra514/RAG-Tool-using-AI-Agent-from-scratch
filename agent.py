import os
import json
from dotenv import load_dotenv
from groq import Groq
from tools import TOOLS,calculator,web_search,doc_search,save_to_memory

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#converts string from LLM "calculator" into actual python function calculator
TOOL_MAP = {
    "calculator": calculator,
    "web_search": web_search,
    "doc_search": doc_search
}

#Think of role as who’s talking and content as what they said.
# Without role, the model wouldn’t know if a line is an instruction, a question, or its own past answer.
#Your conversation list is basically the memory of the chat.
# By starting with:
# {
#   "role": "system",
#   "content": "You are a helpful agent..."
# }
# you’re telling the model:
# “Here are your rules: you have 3 tools, and you must always use one before answering.”
conversation = [
    {
        "role": "system",
        "content": (
            "You are a helpful agent. You have three tools:"
            "doc_search for question about the uploaded document,"
            "web_search for current information,"
            "calculator for math."
            "Always use a tool before answering. Never guess."
        )
    }
]

# Key parameters:
# tools=TOOLS → tells LLM what tools exist
# tool_choice="auto" → LLM decides whether to use a tool
def run(user_input):
    if len(conversation) > 20:
        conversation[1:] = conversation[-18:]
    conversation.append({"role": "user", "content": user_input})

    # detect intent manually
    q = user_input.lower()
    if any(x in q for x in ["calculate", "what is", "compute", "+", "-", "*", "/"]) and any(c.isdigit() for c in q):
        tool_result = calculator(user_input.replace("calculate", "").strip())
        tool_used = "calculator"
    elif any(x in q for x in ["who is", "what is the", "current", "latest", "news", "ceo", "president"]):
        tool_result = web_search(user_input)
        tool_used = "web_search"
    else:
        tool_result = doc_search(user_input)
        tool_used = "doc_search"

    print(f"\n[tool: {tool_used}]")

    messages = conversation[:-1] + [
        {
            "role": "user",
            "content": f"Question: {user_input}\n\nContext from {tool_used}:\n{tool_result}\n\nAnswer based on this context."
        }
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    answer = response.choices[0].message.content
    if len(answer) > 100:
        save_to_memory(f"Q: {user_input}\nA: {answer}")
    conversation.append({"role": "assistant", "content": answer})
    return answer

# This line means: “Only run the following code if this file is executed directly, not imported as a module.”
# It’s a standard Python pattern to separate reusable code (functions) from the script’s main execution.
if __name__ == "__main__":
    while True:
        query = input("\nYou: ".strip())
        if query.lower() == "quit":
            break
        print(f"\nAgent: {run(query)}")
