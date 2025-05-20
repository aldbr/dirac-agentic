import os
from huggingface_hub import login
from smolagents import CodeAgent, DuckDuckGoSearchTool, tool, InferenceClientModel

login(os.environ["HF_TOKEN"])

# Tool to suggest a menu based on the occasion
@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."


agent = CodeAgent(tools=[DuckDuckGoSearchTool(), suggest_menu], model=InferenceClientModel(), additional_authorized_imports=["datetime"])

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
agent.run("Prepare a formal menu for the party.")
