from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model = "llama3.2")

template = """
You are an expert in answering questions about a pizza resturant

Here are some relevant reviews:  {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-----------------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "exit":
        break
    
    reviews = retriever.invoke(question)

    result = chain.invoke({"reviews" : reviews, "question": question})
    print(result)
# from ollama import chat
# from ollama import ChatResponse

# response: ChatResponse = chat(model='llama3.2', messages=[
#   {
#     'role': 'user',
#     'content': 'Who is narendra modi?',
#   },
# ])
# print(response['message']['content'])
# # or access fields directly from the response object
# print(response.message.content)