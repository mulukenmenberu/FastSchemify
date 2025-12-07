"""
Example usage of FastSchemify
"""
from fast_schemify import FastSchemify

# Example 1: Using ORM mode (default)
generate = FastSchemify(type='orm', output='generated_api')
generate.run()

# # Example 2: Using raw SQL queries mode
# generate = FastSchemify(type='query', output='generated_api_query')
# generate.run()

# # Example 3: Step by step
# generate = FastSchemify(type='orm')
# generate.connect()  # Connect to database
# generate.generate()  # Generate the project

