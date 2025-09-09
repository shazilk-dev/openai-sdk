from agents import Agent, InputGuardrail, InputGuardrailTripwireTriggered, Runner, GuardrailFunctionOutput,  enable_verbose_stdout_logging, RunContextWrapper
from config.config import config
from pydantic import BaseModel
import asyncio

enable_verbose_stdout_logging()


class IsMathHomeWork(BaseModel):
    reason: str
    is_math_homework:bool


input_guardrail_agent = Agent(
    name="isMathHomework",
    instructions="Determine if the user's question is related to math homework.",
    output_type=IsMathHomeWork
)

async def is_math_homework(ctx: RunContextWrapper[None], agent: Agent, input: str) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrail_agent, input, run_config=config)
    final_output = result.final_output_as(IsMathHomeWork)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=final_output.is_math_homework,
    )

customer_agent = Agent(
    name="customerServiceAgent",
    instructions="You are a customer service agent, answer the user's questions politely",
     input_guardrails=[
        InputGuardrail(guardrail_function=is_math_homework),
    ],
)


async def main():
    try: 
        user_input = "Can you help me solve this equation: 2x + 3 = ?"
        result = await Runner.run(customer_agent, input=user_input, run_config=config)
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e)

asyncio.run(main())