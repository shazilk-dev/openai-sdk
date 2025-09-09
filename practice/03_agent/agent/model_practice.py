from agents import Agent, ModelSettings
from tools.simple_tools import calculate_area
from hooks.agent_hooks import TestAgHooks


print("\n❄️🔥 Temperature Settings")
print("-" * 30)
    
agent_cold = Agent(
        name="Cold Agent",
        instructions="You are a helpful assistant.",
        model_settings=ModelSettings(temperature=0.1),
    )
    
agent_hot = Agent(
        name="Hot Agent",
        instructions="You are a helpful assistant.",
        model_settings=ModelSettings(temperature=1.9),
    )
    


print("\n🔧 Tool Choice Settings")
print("-" * 30)
    
agent_auto = Agent(
        name="Auto",
        tools=[calculate_area],
        model_settings=ModelSettings(tool_choice="auto"),
    )
    
agent_required = Agent(
        name="Required",
        instructions="you are an helpfull assistant and you must don't use any tool.",
        tools=[calculate_area],
        model_settings=ModelSettings(tool_choice="required"),
        hooks=TestAgHooks(ag_display_name="Required Agent"),
    )

agent_none = Agent(
        name="None",
        tools=[calculate_area],
        model_settings=ModelSettings(tool_choice="none"),
    )
    

print("\n🔧 Top P and penalties Choice Settings")
print("-" * 30)

focused_agent = Agent(
    name="My assistent",
    instructions="you are an helpful assistent",
    model_settings=ModelSettings(
        top_p=0.3,              # Use only top 30% of vocabulary
        # frequency_penalty=0.5,   # Avoid repeating words
        # presence_penalty=0.3 ,    # Encourage new topics
        max_tokens=300

    )
)

unfocused_agent = Agent(
    name="My Assistent",
    instructions="You are an helpful assistent",
    model_settings=ModelSettings(
        max_tokens=300
    )
)
