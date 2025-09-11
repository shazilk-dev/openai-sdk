from agents import Agent, Runner, function_tool, enable_verbose_stdout_logging, ModelSettings, handoff
from config.config import config
from agents.extensions import handoff_filters
import asyncio
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX


print(RECOMMENDED_PROMPT_PREFIX)

triage_agent = Agent(
    name="Triage Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    Your primary job is to diagnose the user's problem.
    If it is about billing, handoff to the Billing Agent.
    If it is about refunds, handoff to the Refund Agent."""
)

enable_verbose_stdout_logging()


@function_tool
def divide(a: int, b: int) -> str:
    """Divides two numbers."""
    # try:
    result = a / b
    return str(result)
    # except ZeroDivisionError:
    #     return "Error: You cannot divide by zero. Please ask for a different number."


agent = Agent(
    name="Math Agent",
    instructions="You are a math assistant. You can perform division operations.",
    tools=[divide],
    model_settings=ModelSettings(
        tool_choice="required"
    )
)



@function_tool
def research_topic(topic: str) -> str:
    """
    Conduct research on a given topic and return a summary of findings.
    """
    return f"Summary of research on {topic}."

@function_tool
def research_intro(topic:str) -> str:
    """" give shrot intro about the given topic """
    return f"intro of research topic: {topic}"


physics_research_agent = Agent(
    name="Physics Research Agent",
    instructions="An agent that conducts research on physics topics and summarizes the findings.",
    handoff_description="Conduct research on a given physics topic and return a summary of findings.",
    tools=[research_topic]
)

islamic_history_research = Agent(
    name="Islamic History Research Agent",
    instructions="An agent that conducts research on Islamic history topics and summarizes the findings.",
    handoff_description="Conduct research on a given Islamic history topic and return a summary of findings.",
    tools=[ research_topic]
)


historical_research_agent = Agent(
    name="Historical Research Agent",
    instructions="your task is to first give an intro about the given research topic and then pass to the relevant research agent for research. if the topic is related to islamic history then handoff to islamic_history_research agent.",
    handoff_description="Conduct historical research on a given topic",
    tools=[research_intro],
    model_settings=ModelSettings(
        tool_choice="required"
    ),
    handoffs=[handoff(islamic_history_research, input_filter=handoff_filters.remove_all_tools)]
)



researcher = Agent(
    name="Researcher",
    instructions="You are a researcher who can delegate tasks to specialized research agents.",
    handoffs=[physics_research_agent, historical_research_agent, ]
)


async def main():
    result = await Runner.run(researcher, "Research about the golden age of islam and summarize the findings.",  run_config=config)

    print(result.final_output)


    # result = await Runner.run(agent, "Divide 10 by 0",   run_config=config)
    # print(result.final_output)


asyncio.run(main())