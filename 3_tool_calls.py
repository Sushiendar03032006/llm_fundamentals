from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI()

# ---- 1. The problem - model doesn't know real time data ----
result = client.responses.create(
    model="gpt-4o",
    input="What is the current weather in Bangalore?"
)

print(result.output_text)
# Output: "I don't have access to real time weather data..."

# ---- 2. Define a function the model can use ----
def get_weather(city: str):
    # In real life this would call a weather API
    # For now we're faking it
    return f"The weather in {city} is 28 degrees and sunny."

# ---- 3. Define the tool - this is how you describe your function to the model ----
tools = [
    {
        "type": "function",
        "name": "get_weather",                        # moved outside "function" block
        "description": "Get the current weather for a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get weather for"
                }
            },
            "required": ["city"]
        }
    }
]

# ---- 4. The full loop ----

# Step 1 - send the question + tools to the model
result = client.responses.create(
    model="gpt-4o",
    input="What is the current weather in Bangalore?",
    tools=tools
)

# Step 2 - model doesn't answer, it tells us to call a function
print(result.output)
# Notice status is 'requires_action' not 'completed'
print(result.status)

# Step 3 - we extract what function the model wants to call and with what arguments
tool_call = result.output[0]
function_name = tool_call.name
function_args = json.loads(tool_call.arguments)

print(f"Model wants to call: {function_name}")
print(f"With arguments: {function_args}")

# Step 4 - we actually run the function ourselves
function_result = get_weather(function_args["city"])
print(f"Function returned: {function_result}")

# Step 5 - send everything back to the model so it can give the final answer
final_result = client.responses.create(
    model="gpt-4o",
    previous_response_id=result.id,        # replaces manually appending messages
    input=[{
        "type": "function_call_output",    # replaces "role": "tool"
        "call_id": tool_call.call_id,
        "output": function_result
    }],
    tools=tools
)

print(final_result.output_text)
# Output: "The current weather in Bangalore is 28 degrees and sunny."