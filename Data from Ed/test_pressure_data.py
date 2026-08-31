# AI generated

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
# import os


# # go to current directory
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
 

# 1. Load the saved compressed file
windCase = '10B'
number_of_closest_rollers = 7
angle_offset = 0#np.pi/6
saved_data = np.load(f"inner_ring_segment_pressure_{windCase}_closest_rollers_{number_of_closest_rollers}_angle_offset_{round(angle_offset, 2)}.npz")

time = saved_data['time']
closest_idx = saved_data['roller_indices']
rel_angles = saved_data['relative_angle_rad']
P_max = saved_data['P_max_inner_MPa']
a_in = saved_data['a_in_mm']
b_in = saved_data['b_in_mm']

# 2. Setup geometry
R_inner = 500.0  # Bearing inner ring radius in mm
theta_plot = np.linspace(-0.4, 0.4, 1000)
x_plot = theta_plot * R_inner  # Arc length coordinate (mm)

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(x_plot[0], x_plot[-1])
ax.set_ylim(0, np.max(P_max) * 1.2 if np.max(P_max) > 0 else 3000)
ax.set_xlabel('Arc Length from Segment Center (mm)', fontsize=11)
ax.set_ylabel('Hertzian Contact Pressure (MPa)', fontsize=11)
ax.set_title(f'Dynamic Pressure Profile (Wind Case {windCase})', fontsize=13, pad=12)
ax.grid(True, linestyle='--', alpha=0.5)

line, = ax.plot([], [], lw=2.5, color='#1f77b4')
time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def update(frame):
    pressure_total = np.zeros_like(x_plot)
    
    for r in range(number_of_closest_rollers):
        x_c = rel_angles[frame, r] * R_inner
        b = b_in[frame, r]
        p0 = P_max[frame, r]
        
        in_contact = np.abs(x_plot - x_c) <= b
        pressure_total[in_contact] += p0 * np.sqrt(1 - ((x_plot[in_contact] - x_c) / b)**2)
    
    line.set_data(x_plot, pressure_total)
    time_text.set_text(f'Time: {time[frame]:.2f} s')
    return line, time_text

# Create animation object
anim = FuncAnimation(fig, update, init_func=init, frames=len(time), interval=40, blit=True)

# Option A: Save it as a GIF to view externally
output_gif = f'pressure_animation_{windCase}_closest_rollers_{number_of_closest_rollers}_angle_offset_{round(angle_offset, 2)}.gif'
anim.save(output_gif, writer=PillowWriter(fps=25))
print(f"Animation saved successfully as {output_gif}")

# Option B: Display interactively in a window
plt.show()