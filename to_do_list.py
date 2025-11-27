from langchain.chat_models import init_chat_model
from langchain.agents.middleware import TodoListMiddleware
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
import pprint
load_dotenv()

my_llm = init_chat_model(
    model = "gpt-4o-mini",
    model_provider='openai'
    )


@tool#(parse_docstring=True)
def add(a: float, b: float)->float:
    """
    Adds two floating point numbers
    Args:
        a : First floating point number
        b : Second floating point number
    Returns:
        a+b
    """

    return a+b

@tool#(parse_docstring=True)
def subtract(a:float, b: float) -> float:
    """
    Performs subtraction of two floating point numbers.

    Args:
        a : First floating point number
        b : Second floating point number
    Returns:
        a-b

    """

    return a-b

@tool#(parse_docstring=True)
def multiple(a: float, b: float) -> float:
    """
    Performs the product of two floating point numbers.
    
    Args:
        a : First floating point number
        b : Second floating point number
    Returns:
        a*b
    """

    return a*b


my_agent = create_agent(
    model=my_llm,
    tools = [],
    middleware=[TodoListMiddleware()],
    system_prompt="You are an arithmetic agent. You have access to add, subtract, and multiplication tools "
)

response = my_agent.invoke({"messages": [{"role": "human", "content" : "what is 2*3+4"}]})
# pprint.pprint(response)
for message in response:
    message.pretty_print()