from pydantic import BaseModel
from config.config import config
from agents import Agent, Runner, RunContextWrapper, InputGuardrailTripwireTriggered, input_guardrail, enable_verbose_stdout_logging, GuardrailFunctionOutput,  TResponseInputItem
import asyncio
from hooks.agent_hooks import TestAgHooks

enable_verbose_stdout_logging()

class MathHomeworkOutput(BaseModel):
    is_math_homework: bool
    reasoning: str


guardrail_agent: Agent = Agent(
    name="Homework Police",
    instructions="Check if the user is asking you to do their math homework.",
    output_type=MathHomeworkOutput,
    hooks=TestAgHooks("Guardrail Agent"),

)

@input_guardrail
async def math_guardrail(ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context, run_config=config)
    return GuardrailFunctionOutput(
        output_info=result.final_output.reasoning,
        tripwire_triggered=not result.final_output.is_math_homework
    )

Math_tutor = Agent(  
    name="Math Tutor",
    instructions="You are a helpful math tutor that help with math related queries",
    input_guardrails=[math_guardrail],  # Attach our guardrail
    hooks=TestAgHooks("Math_tutor")
)


async def main():
    try:
        # This should trigger the guardrail
        result = await Runner.run(Math_tutor, "Tell me about pakistan", run_config = config)
        print(result.final_output)    
        # print(result.context_wrapper.usage)
    except InputGuardrailTripwireTriggered:
        print("Guardrail Triggered")
        # print(result.context_wrapper.usage)
        # Handle appropriately - maybe send a polite rejection message

asyncio.run(main())