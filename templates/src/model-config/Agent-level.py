import asyncio
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner


#Reference: https://ai.google.dev/gemini-api/docs/openai
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

async def main():
    # This agent will use the custom LLM provider
    agent = Agent(
        name="Assistant",
        instructions="You only respond in urdu.",
        model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=client),
    )

    result = await Runner.run(
        agent,
        "I am learning Agentic AI with Panaversity Community",
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
     