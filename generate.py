"""
Example usage of FastSchema
"""
from fast_schema import FastSchema

# Example 1: Using ORM mode (default)
generate = FastSchema(type='sql', output='generated_api')
generate.run()

# # Example 2: Using raw SQL queries mode
# generate = FastSchema(type='query', output='generated_api_query')
# generate.run()

# # Example 3: Step by step
# generate = FastSchema(type='orm')
# generate.connect()  # Connect to database
# generate.generate()  # Generate the project

