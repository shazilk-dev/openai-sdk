
import asyncio
from runners.first_agent_runner import first_agent
from runners.model_practice_runner import model_practice_runner
## Model setting: Agents imports


async def main():

    # await first_agent()
    await model_practice_runner()


if __name__ == "__main__":
    asyncio.run(main())