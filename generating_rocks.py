import random
import math

def generate_random_rocks(num_rocks, min_x, max_x, min_y, max_y, min_z, max_z, model_names):
    rocks = []
    for i in range(num_rocks):
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        z = random.uniform(min_z, max_z)
        roll = random.uniform(0, 2 * math.pi)
        pitch = random.uniform(0, 2 * math.pi)
        yaw = random.uniform(0, 2 * math.pi)
        model = random.choice(model_names)
        
        rock = f"""
  <model name="rock{i+1}">
    <include>
      <uri>model://{model}</uri>
    </include>
    <pose>{x:.2f} {y:.2f} {z:.2f} {roll:.2f} {pitch:.2f} {yaw:.2f}</pose>
  </model>
"""
        rocks.append(rock)
    return rocks

# Example usage
num_rocks = 25
min_x, max_x = -75, 75
min_y, max_y = -75, 75
min_z, max_z = 0, 2
model_names = ["'Large Rock Fall'", "'Small Rock Fall'", "'Medium Rock Fall'"]  # Add your model folder names here

random_rocks = generate_random_rocks(num_rocks, min_x, max_x, min_y, max_y, min_z, max_z, model_names)

# Print the generated XML for each rock
for rock in random_rocks:
    print(rock)