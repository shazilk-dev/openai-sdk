from agents import Agent, Runner, enable_verbose_stdout_logging
from config.config import config
import asyncio

enable_verbose_stdout_logging()


booking_agent = Agent(
    name="Booking expert agent",
    instructions=(
        "Handle all booking-related questions and requests. "
        "Do not handle refund requests."
    ),
)
refund_agent = Agent(
    name="Refund expert agent",
    instructions=(
        "Handle all refund-related questions and requests. "
        "Do not handle booking requests."
    ),
)

customer_facing_agent = Agent(
    name="Customer-facing agent",
    instructions=(
        "Handle all direct user communication. "
        "Call the relevant tools when specialized expertise is needed."
    ),
    tools=[
        booking_agent.as_tool(
            tool_name="booking_expert",
            tool_description="Handles booking questions and requests.",
        ),
        refund_agent.as_tool(
            tool_name="refund_expert",
            tool_description="Handles refund questions and requests.",
        )
    ],
)


async def main():
    result = await Runner.run(
        customer_facing_agent,
        "I need to change my booking and get a refund for my last trip.",
        run_config=config,
    )

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())