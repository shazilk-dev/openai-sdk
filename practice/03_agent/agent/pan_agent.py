import asyncio
import random
from agents import Agent, Runner, function_tool, enable_verbose_stdout_logging
from config.config import config
from openai.types.responses import ResponseTextDeltaEvent


# enable_verbose_stdout_logging()

# agent: Agent = Agent(
#     name="Assistant",
#     instructions="You are a helpful assistant",
# )

# result = Runner.run_sync(agent, "Write a haiko about recursion in programming", run_config=config)
# print(result.final_output)

# print(result)

@function_tool
def how_many_jokes():
    """Give the quanitiy of jokes """
    return random.randint(1, 10)
    


async def main():
    agent: Agent = Agent(
        name="joker agent",
                instructions="You are a helpful assistant. First, determine how many jokes to tell, then provide jokes.",
        tools=[how_many_jokes]

    )

    result = Runner.run_streamed(agent, "Hello", run_config=config)
    print(f'Resut: {result}')

    # item = result.stream_events()
    # print(item)
    async for event in result.stream_events():
        print(f'\n [EVENT]: {event}\n')



asyncio.run(main())



















# If you want to run this file directly, use:
# python -m agent.pan_agent


