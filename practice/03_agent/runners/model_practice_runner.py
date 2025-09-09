from agents import  Runner
from config.config import config
from agent.model_practice import agent_cold, agent_hot, agent_auto, agent_required, agent_none, focused_agent, unfocused_agent



async def model_practice_runner():
     ## Model setting 
#     question = "What's the area of a 5x3 rectangle?"
    
#     print("Cold Agent (Temperature = 0.1):")
#     result_cold = await Runner.run(agent_cold, question, run_config=config)
#     print(result_cold.final_output)
    
#     print("\nHot Agent (Temperature = 1.9):")
#     result_hot = await Runner.run(agent_hot, question, run_config=config)
#     print(result_hot.final_output)
    
#     print("\n💡 Notice: Cold = focused, Hot = creative")
#     print("📝 Note: Gemini temperature range extends to 2.0")

    # 🎯 Example 2: Tool Choice

    question = "What's the area of a 5x3 rectangle?"
    
    # print("Auto Tool Choice:")
    # result_auto = await Runner.run(agent_auto, question, run_config=config)
    # print(result_auto.final_output)
    
    print("\nRequired Tool Choice:")
    result_required = await Runner.run(agent_required, question, run_config=config)
    print(result_required.final_output)
  

    # print("\nNone Tool Choice:")
    # result_none = await Runner.run(agent_none, question, run_config=config)
    # print(result_none.final_output)
    
    # print("\n💡 Notice: Auto = decides, Required = must use tool")

#     print("\nFocused Agent (top_p = 0.3, frequency_penalty = 0.5, presence_penalty = 0.3):")
#     result_focused = await Runner.run(focused_agent, "Tell me a short story about a dragon and a knight.", run_config=config)
#     print(result_focused.final_output)

#     print("\nunfocused Agent :")
#     result_unfocused = await Runner.run(unfocused_agent, "Tell me a short story about a dragon and a knight.", run_config=config)
#     print(result_unfocused.final_output)
