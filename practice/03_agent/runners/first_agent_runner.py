from agents import Runner
from agent.first_agent import simple_agent
from config.config import config

async def first_agent():
    result = await Runner.run(simple_agent, "give me detail about weather", run_config=config)
    print('\n\n --------------------- Final Output: --------------------- \n\n')
    print(result.final_output)