from agents import Agent, Runner, function_tool, enable_verbose_stdout_logging, ModelSettings
from config.config import config
import asyncio

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


physics_research_agent = Agent(
    name="Physics Research Agent",
    instructions="An agent that conducts research on physics topics and summarizes the findings.",
    handoff_description="Conduct research on a given physics topic and return a summary of findings.",
    tools=[research_topic]
)

historical_research_agent = Agent(
    name="Historical Research Agent",
    instructions="An agent that conducts historical research on a given topic and summarizes the findings.",
    handoff_description="Conduct historical research on a given topic and return a summary of findings.",
    tools=[research_topic]
)

researcher = Agent(
    name="Researcher",
    instructions="You are a researcher who can delegate tasks to specialized research agents.",
    handoffs=[physics_research_agent, historical_research_agent]
)


async def main():
    # result = await Runner.run(researcher, "Research about the golden age of islam and summarize the findings.", max_turns=3 ,  run_config=config)

    # print(result.final_output)


    result = await Runner.run(agent, "Divide 10 by 0",   run_config=config)
    print(result.final_output)


asyncio.run(main())