from agents import Agent,  ModelSettings, Runner
from config.config import config
from hooks.agent_hooks import TestAgHooks
from tools.simple_tools import fetch_weather
import asyncio
import rich

partner_agent: Agent = Agent(
    name="Partner Agent",
    instructions="You are a partner agent, when you will get the handoffs from simple agent then  you provide the final response.. with that weather information providing relevant information about what precautions to do in that weather in just 3 short points.",
    handoff_description='provide weather related information',
    # hooks=TestAgHooks(ag_display_name="Partner-Agent"),
    model_settings=ModelSettings(
        temperature=0.4,
        max_retries=3,
        max_tokens=200
    )

)

simple_agent: Agent = Agent(
    name="My Assistant",
    instructions="You are an helpful assisant, if user ask about weather then call the fetch weather tool if user mention the city name then pass that city name otherwise use default city name and give output and then handoffs to partner. ",
    # hooks=TestAgHooks(ag_display_name="Starting_lead Agent"),
    tools=[fetch_weather],
    handoffs=[partner_agent]
  
)


async def main():
    result = await Runner.run(simple_agent, "What's the weather like in New York?",  run_config=config)
    rich.print(result.final_output)


    rich.print(result.last_agent.name)
    rich.print(result.to_input_list())

asyncio.run(main())