def parse_agent_response(response):
    # Create a list to hold the desired format
    formatted_data = []

    # Iterate through the data and format it
    for action, description in response["intermediate_steps"]:
        formatted_data.append(
            [[action.tool, action.tool_input, action.log], description]
        )

    # Serialize the formatted data to JSON with indentation
    json_data = json.dumps(formatted_data, indent=2)
    return json_data
