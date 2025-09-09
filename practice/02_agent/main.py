from config import config
from agents import Agent, Runner


partner_agent: Agent = Agent(
    name="My Partner",
    instructions="you are a helpful Partner",
    handoff_description="help the simple agent"
)


simple_agent: Agent = Agent(
    name= 'My Assistant',
    instructions='you are an helpful Assistant when asked about partner handoffs to partner agent',
    handoffs=[partner_agent]
)


def main():
    result = Runner.run_sync(
        simple_agent,
        'i am learning making ai agents. give me atleast 5 short points to learn in the best way',
        run_config=config
    )
    print(result.final_output)

if __name__ == '__main__':
    main()




