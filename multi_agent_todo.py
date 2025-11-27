from langchain.chat_models import init_chat_model
from langchain.agents.middleware import TodoListMiddleware
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
import pprint
load_dotenv()

all_calls = []

supervisor_llm = init_chat_model(
    model = "gpt-4o-mini",
    model_provider='openai',
    )

rag_llm = init_chat_model(
    model = "gpt-4o-mini",
    model_provider='openai',
)

calc_llm = init_chat_model(
    model = 'gpt-4o-mini',
    model_provider='openai',
)




@tool#(parse_docstring=True)
def addition(a: float, b: float)->float:
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
def subtraction(a:float, b: float) -> float:
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
def multiplication(a: float, b: float) -> float:
    """
    Performs the product of two floating point numbers.
    
    Args:
        a : First floating point number
        b : Second floating point number
    Returns:
        a*b
    """

    return a*b

@tool#(parse_docstring=True)
def division(a: float, b: float) -> float:
    """
    Performs the division of two floating point numbers.
    
    Args:
        a : First floating point number
        b : Second floating point number
    Returns:
        a/b
    """

    return a/b

calc_agent = create_agent(
    model=calc_llm,
    tools = [addition, subtraction, multiplication, division],
    middleware=[TodoListMiddleware()],
    system_prompt="You are an arithmetic agent. You have access to addition, subtraction, multiplication, and division tools "
)



@tool
def retreive_augment_context(query: str) -> str:
    """
    Retreives policy information about TCS company.
    
    Args:
        query : Query about the company

    Returns:
        Returns the data retreived from the vector db to answer the query and augments the query.
    """
    return f"{query} : TCS has very strict policies about personal information"



@tool
def generate_respond() -> str:
    """
    Generates final response for the query.
    Returns:
        Returns the final response for the query.
    """
    return "TCS has very strict policies about personal information"


rag_agent = create_agent(
    model=rag_llm,
    tools = [retreive_augment_context, generate_respond],
    middleware=[TodoListMiddleware()],
    system_prompt="""You are a RAG agent. Your task is to answer user queries that are relates to TCS policies.
    Use the below tools to respond to all questions related to TCS policies.
    Do not hallucinate, just invoke the tools to get the information regarding all TCS policies.
    You have access to retreive_augment_context, and generate_respond tools 
    Do not add extra text other than what tools provide"""
)

@tool
def calc_agent_tool(query: str):
    """
    Capable of performing expression evaluation that contain addition, subtraction, multiplication, and division operations

    Args:
        query: Contains the expression to evaluate.

    Returns:
        The final value after evaluation
    """
    calc_resp =  calc_agent.invoke(
        {
            "messages": [{
                "role": "human",
                "content" : query
                }
            ]
        }
    )
    all_calls.append(calc_resp)
    return calc_resp['messages'][-1].content

@tool
def rag_policy_agent_tool(query: str):
    """
    Capable of generating responses for any TCS related Policies.

    Args:
        query: Contains the query related to TCS Policies.

    Returns:
        The response to the query related to TCS Policies
    """
    rag_resp =  rag_agent.invoke(
        {
            "messages": [{
                "role": "human",
                "content" : query
                }
            ]
        }
    )
    all_calls.append(rag_resp)
    return rag_resp['messages'][-1].content

supervisor_agent = create_agent(
    model=supervisor_llm,
    tools=[rag_policy_agent_tool, calc_agent_tool],
    middleware=[TodoListMiddleware()],
    system_prompt=""" You are an expert agent that can repond to queries related to expression evaluation and
    TCS policies. You have access to "calc_agent_tool", and "rag_policy_agent_tool" to provide responses to
    expression evaluation and TCS policies respectively.

    Donot hallicunate and respond.
    Respond only with the information obtained from the tools available in hand. 

"""
)


# response = supervisor_agent.invoke({"messages": [{"role": "human", "content": "query : Answer about TCS policies?"}]})
response = supervisor_agent.invoke({"messages": [{"role": "human", "content": "query : 2*3-2?"}]})

print(response)


# response = rag_agent.invoke({"messages": [{"role": "human", "content": "query = Answer about TCS policies?"}]})

# response['messages'][-1].content

# response = my_agent.invoke({"messages": [{"role": "human", "content" : "what is 2*3+4"}]})
# pprint.pprint(response)
# for message in response:
#     message.pretty_print()