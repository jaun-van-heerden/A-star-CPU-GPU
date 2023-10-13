import pstats
import glob
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

# Read profile files and populate data
profile_files = glob.glob('profiles/*.prof')
data = {}

for file in profile_files:
    _, type_, num = file.split('_')
    num = num.split('.')[0]
    key = f"{type_}_{num}"
    stats = pstats.Stats(file)
    total_time = stats.total_tt
    data[key] = total_time

# Matplotlib Line Plot
types = ['vec', 'seq', 'gpu']
nums = [1, 2, 4, 8, 16, 32, 64]

for type_ in types:
    y_values = [data[f"{type_}_{num}"] for num in nums]
    plt.plot(nums, y_values, label=f'Type: {type_}')

plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Step Size (log scale)')
plt.ylabel('Total Time (log scale)')
plt.title('Profiling Results')
plt.legend()
plt.show()





import seaborn as sns

# Create a Pandas DataFrame from the data dictionary
import pandas as pd
df = pd.DataFrame(list(data.items()), columns=['Type_Step', 'Time'])
df['Type'] = df['Type_Step'].apply(lambda x: x.split('_')[0])
df['Step'] = df['Type_Step'].apply(lambda x: int(x.split('_')[1]))

# Create the heatmap
plt.figure(figsize=(10, 8))
#sns.heatmap(df.pivot("Step", "Type", "Time"), annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Total Time'})
sns.heatmap(df.pivot(index="Step", columns="Type", values="Time"), annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Total Time'})

plt.title('Profiling Heatmap')
plt.show()



for type_ in types:
    y_values = [data[f"{type_}_{num}"] for num in nums]
    plt.plot(nums, y_values, label=f'Type: {type_}')

# Remove the log scale lines
# plt.xscale('log', base=2)
# plt.yscale('log')
plt.xlabel('Step Size')
plt.ylabel('Total Time')
plt.title('Profiling Results')
plt.legend()
plt.show()





# # Plotly 3D Surface Plot
# types = []
# nums = []
# times = []

# # Populate lists
# for key, value in data.items():
#     type_, num = key.split('_')
#     types.append(type_)
#     nums.append(int(num))
#     times.append(value)

# # Convert lists to NumPy arrays and reshape for 3D surface plot
# types = np.array(types)
# nums = np.array(nums)
# times = np.array(times).reshape((len(types)//len(nums), len(nums)))

# # Create the plot
# fig = go.Figure(data=[go.Surface(z=times, x=types, y=nums)])
# fig.update_layout(
#     title='Profiling Data Across Different Conditions',
#     scene=dict(
#         xaxis_title='Type',
#         yaxis_title='Step Size',
#         zaxis_title='Total Time'
#     ),
#     margin=dict(l=0, r=0, b=0, t=40),
#     scene_camera=dict(eye=dict(x=1.5, y=0.1, z=0.8)),
#     template='plotly_dark'
# )

# fig.show()


