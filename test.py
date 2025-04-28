import matplotlib.pyplot as plt
import re

# Read the data from the provided text
file_path = "test.txt"
with open(file_path, "r") as file:
    lines = file.readlines()

# Extract iterations and win rates using regex
iterations = []
win_rates = []

for line in lines:
    match = re.search(r"#(\d+)\s+.*\| WinRate: (\d+\.\d+)", line)
    if match:
        iterations.append(int(match.group(1)))
        win_rates.append(float(match.group(2)))

# Plot the graph
plt.figure(figsize=(10, 5))
plt.plot(iterations, win_rates, marker="o", linestyle="-", color="b", label="Win Rate")

# Labels and title
plt.xlabel("Iterations")
plt.ylabel("Win Rate")
plt.title("Win Rate vs. Iterations Over Time")
plt.legend()
plt.grid()

# Show the plot
plt.show()