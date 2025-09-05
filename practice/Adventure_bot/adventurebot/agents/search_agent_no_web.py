from agents import Agent, handoff
from ..tools.context_tools import update_child_threshold_status
from ..models import TripContext, CHILD_AGE_THRESHOLD, ActivityResult, SearchResult  # Import models from ..models

PROMPT_WITHOUT_WEB_SEARCH = f"""You research and find suitable activities for a trip based on provided details.

Given the trip details (location, dates, participant ages) and weather information:

1. **Check for young children:** Use the `update_child_threshold_status` tool to determine if any participant is under {CHILD_AGE_THRESHOLD} years old. If the tool indicates the threshold is met, **HANDOFF** the task to the 'Kid-Friendly Activity Agent'. Provide the original trip details and weather summary in the handoff notes.

2. **If no young children (threshold not met):**
   a. Based on your knowledge, brainstorm 5-7 popular and suitable activities for the destination.
   b. Consider the weather conditions, participant ages, and local attractions.
   c. For each activity, provide structured information including:
      - Name and description
      - Typical location within the destination
      - Estimated age appropriateness
      - Approximate price range
      - Duration
      - Weather dependency
   d. Compile a list of structured ActivityResult objects.
   e. Provide a concise summary of your recommendations.
   f. Return the results in the SearchResult format.

**Important:** Prioritize the child threshold check using the dedicated tool before proceeding with recommendations.
**Note:** Use your existing knowledge about popular destinations and activities since web search is not available."""


def create_activity_search_agent_no_web(model: str = "litellm/gemini/gemini-2.0-flash-exp") -> Agent[TripContext]:
    """Create an agent that searches for activities without web search tool."""
    from .kid_friendly_agent_no_web import create_kid_friendly_activity_agent_no_web  # Import from the correct file
    kid_friendly_agent = create_kid_friendly_activity_agent_no_web(model=model)
    return Agent[TripContext](
        name="Activity Search Agent",
        instructions=PROMPT_WITHOUT_WEB_SEARCH,
        output_type=SearchResult,
        model=model,
        tools=[update_child_threshold_status],
        handoffs=[handoff(kid_friendly_agent)],
    )
