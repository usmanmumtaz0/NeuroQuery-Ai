from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.chains.router import MultiPromptChain
from dotenv import load_dotenv
import os


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")  # Read API key from .env

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    api_key=api_key  # Pass the key securely
)

news_t = PromptTemplate.from_template("You are a news expert. Answer: {input}")
fin_t = PromptTemplate.from_template("You are a finance expert. Answer: {input}")
tech_t = PromptTemplate.from_template("You are a tech expert. Answer: {input}")

news_c = LLMChain(llm=llm, prompt=news_t)
fin_c = LLMChain(llm=llm, prompt=fin_t)
tech_c = LLMChain(llm=llm, prompt=tech_t)

router = MultiPromptChain.from_prompts(
    llm=llm,
    prompt_infos=[
        {"name": "news", "description": "news related questions", "prompt_template": "You are a news expert. {input}"},
        {"name": "finance", "description": "finance related questions", "prompt_template": "You are a finance expert. {input}"},
        {"name": "technology", "description": "technology related questions", "prompt_template": "You are a tech expert. {input}"}
    ]
)

print("AI Research Assistant  🚀")

while True:
    q = input("\nAsk (type 'exit' to quit): ")

    if q.lower() == "exit":
        print("Goodbye 👋")
        break

    try:
        response = router.run(q)
        print("\nAnswer:", response)
    except Exception as e:
        print("Error:", e)