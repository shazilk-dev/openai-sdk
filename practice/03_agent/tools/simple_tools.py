from agents import function_tool


@function_tool(is_enabled=True)
def fetch_weather(city:str) -> str:
    """ use to fetch weather """
    return f'Weather of {city} is cloudy'


@function_tool
def calculate_area(length: float, width: float) -> str:
    """Calculate the area of a rectangle."""
    area = length * width
    return f"Area = {length} × {width} = {area} square units"