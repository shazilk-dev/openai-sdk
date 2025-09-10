from agents import Agent, Runner, RunContextWrapper, handoff, enable_verbose_stdout_logging
import asyncio
from config.config import config
from pydantic import BaseModel

enable_verbose_stdout_logging()

def log_handoff_event(ctx: RunContextWrapper):
    print(f"HANDOFF INITIATED: Transferring to the Escalation Agent by {ctx.context.user_name}")

class UserContext(BaseModel):
        user_id: str
        user_name: str

sir_Ameen = Agent(
    name="Python Teacher",
    instructions="You are a python professor. you need to give 2 important points for learning python",
    handoff_description="use this agent for python query",
    
)

sir_hamza = Agent(
    name="Agentic-ai teacher",
    instructions="You are an agentic ai professor. you need to give 2 short points for learning  agentic-ai",
    handoff_description="use this agent for agentic-ai query"
)

custom_handoff = handoff(
    agent=sir_Ameen,
    tool_name_override="escalate_to_python_specialist",
    tool_description_override="Use this for python issues.",
    on_handoff=log_handoff_event,
)

sir_zia = Agent[UserContext](
    name="Head_of_Learning",
    instructions=" your goal is to transfer control to the relatable agent based on the user query",
    handoffs=[custom_handoff]
)



user = UserContext(user_id="abc", user_name="Shazil")



async def main():
    result = await Runner.run(sir_zia, "give me  short tips about learning python", run_config=config, context=user)
    print(result.final_output)


asyncio.run(main())    



