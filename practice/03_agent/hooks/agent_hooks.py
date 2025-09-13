from agents import Agent, AgentHooks, RunContextWrapper
from typing import Any
import rich
class TestAgHooks(AgentHooks):
    def __init__(self, ag_display_name):
        self.event_counter = 0
        self.ag_display_name = ag_display_name

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        rich.print(f"\n --- {self.ag_display_name} : {self.event_counter}: \n Agent {agent.name} started. \n Usage: {context.usage}")
        rich.print(f"🕘 Agent \"{agent.name}\" is now in charge of handling the task")


    async def on_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        self.event_counter += 1
        rich.print(f"\n\n --- \"{self.ag_display_name}\" : {self.event_counter}: \n Agent {agent.name} ended.\n Usage: {context.usage}, \nOutput: {output}")

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        self.event_counter += 1
        rich.print(f"\n\n --- 📞 \"{self.ag_display_name}\" : {self.event_counter}:\n Agent {agent.name} is asking the AI for help with: \n {input_items}")

    async def on_llm_end(self, context, agent, response):
        self.event_counter += 1
        rich.print(f"\n\n --- \"{self.ag_display_name}\" : {self.event_counter}: \n 🧠✨ Agent {agent.name} got AI response: \n {response}")

    async def on_tool_start(self, context, agent, tool):
        self.event_counter += 1
        print(f"\n\n --- \"{self.ag_display_name}\" : {self.event_counter}: \n 🔨 Agent {agent.name} is using tool: {tool.name}")

    async def on_tool_end(self, context, agent, tool, result):
        self.event_counter += 1
        print(f"\n\n --- \"{self.ag_display_name}\" : {self.event_counter}: \n ✅🔨 Agent {agent.name} finished using {tool.name}. \n Result: {result}")

    async def on_handoff(self, context, agent, source):
        self.event_counter += 1
        print(f"\n\n --- \"{self.ag_display_name}\" : {self.event_counter}: \n Agent {agent.name} handoff. \n Usage: {context.usage}, \n handoff from Source: {source.name}")