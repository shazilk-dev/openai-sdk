from agents import AsyncOpenAI, OpenAIChatCompletionsModel;
from agents.run import RunConfig
import os
from dotenv import load_dotenv
load_dotenv()


external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=os.getenv('GEMINI_API_KEY'),
    base_url=os.getenv('BASE_URL')
)

model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model=os.getenv('GEMINI_MODEL'),
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)