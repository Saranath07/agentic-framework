# batch mode
    
# Ouptut of the first agent is a list and then use this list to get another list using another agent

# Multiprocessing 

# Use the key-value agent
result = kv_agent.invoke(custom_input)
print("Key-Value Parsing Result:")
for key, value in result.items():
    print(f"{key}: {value}")

print("\n" + "-" * 50 + "\n")

# Example of batch processing with a list of people
people = ["Virat kohli", "Naga Chaitanya", "Mukesh Ambani"]
prompt_template = "What is the father's name of {person} and his {surname}?"

my_dict = {
    "person" : ["Virat kohli", "Naga Chaitanya", "Mukesh Ambani"],
    "surname" : ["Kohli", "Akkineni", "Ambani"]
}

# There could be combinations
# 1. All combinations
# 2. One one mapping
# 3. random sampling

Given a single api, send multiple requests to it -> capped by rate limit of api

Solution : Take multiple APIs -> Each api has k limit, nk (n : # apis)
