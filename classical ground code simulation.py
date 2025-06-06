import numpy as np
from scipy.stats import entropy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics import Color, Line, Rectangle
from kivy.clock import Clock
from kivy.properties import NumericProperty, ListProperty, StringProperty
import random
import json
import time
import os
from kivy.utils import platform
import tempfile

class GroundingWidget(BoxLayout):
    frame = NumericProperty(0)
    input_load_data = ListProperty([])
    sys_load_data = ListProperty([])
    temp_data = ListProperty([])
    entropy_data = ListProperty([])
    energy_data = ListProperty([])
    efficiency_data = ListProperty([])
    status_text = StringProperty("Simulation Running")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Simulation parameters
        self.surge_load = 10000  # Surge load (tasks/s)
        self.base_load = 100  # Normal load (tasks/s)
        self.ground_factor = 0.1  # Grounding efficiency
        self.damping = 0.01  # Damping factor
        self.t_max = 0.01  # Simulation time per cycle (s)
        self.dt = 0.00005  # Time step
        self.t_start = 0.001  # Surge start time (s)
        # Thermal parameters
        self.thermal_mass = 0.05  # J/°C
        self.heat_coeff = 0.001  # W per task/s
        self.dissipation_coeff = 0.1  # W/°C
        self.initial_temp = 25  # °C
        self.ambient_temp = 25  # °C
        self.max_load = 1e4  # Cap load
        self.max_temp = 100  # Max temp (°C)
        # Entropy parameters
        self.n_bins = 50  # Bins for histogram
        self.window_size = 100  # Sliding window
        # New: Energy and efficiency
        self.energy = 0  # Total energy (J)
        # Device simulation
        self.use_device = False  # Toggle for device data
        self.device_load = self.base_load
        # Cycle management
        self.cycle_count = 0
        self.max_cycles = 3  # Number of surge cycles
        # Initialize results file
        if platform == 'android':
            self.results_file = os.path.join(tempfile.gettempdir(), f"simulation_results_{int(time.time())}.json")
        else:
            self.results_file = f"simulation_results_{int(time.time())}.json"
        self.results_data = []  # Store all results for JSON
        # Initialize UI
        self.orientation = 'vertical'
        self.label = Label(text="[b]Status:[/b] [color=000000]Simulation Running[/color]", markup=True, size_hint=(1, 0.1))
        self.add_widget(self.label)
        self.plot_widget = BoxLayout(size_hint=(1, 0.8))
        self.add_widget(self.plot_widget)
        self.reset_button = Button(text="Reset Simulation", size_hint=(1, 0.1))
        self.reset_button.bind(on_press=self.reset_simulation)
        self.add_widget(self.reset_button)
        # Initialize simulation
        self.initialize_simulation()
        # Start animation
        Clock.schedule_interval(self.update, 1/60)

    def save_header(self):
        # Initialize an empty JSON file or prepare the structure
        try:
            self.results_data = []  # Reset results data
            with open(self.results_file, 'w') as f:
                json.dump(self.results_data, f, indent=4)
        except Exception as e:
            self.status_text = f"Error initializing JSON file: {str(e)}"

    def initialize_simulation(self):
        # Initialize or reset simulation arrays
        self.time = np.arange(0, self.t_max, self.dt)
        self.input_load = np.where(self.time >= self.t_start, self.surge_load, self.base_load)
        self.sys_load = np.zeros_like(self.time)
        self.temp = np.zeros_like(self.time)
        self.entropy_vals = np.zeros_like(self.time)
        self.energy_vals = np.zeros_like(self.time)
        self.efficiency_vals = np.zeros_like(self.time)
        self.temp[0] = self.initial_temp
        self.energy = 0
        # Simulate
        for i in range(1, len(self.time)):
            # Device data simulation (replace with actual device input)
            if self.use_device:
                self.device_load = self.base_load + random.uniform(-50, 50)  # Mock device fluctuation
                self.input_load[i] = self.device_load if self.time[i] < self.t_start else self.surge_load
            # System dynamics
            excess_load = self.input_load[i] - self.base_load
            grounded_load = excess_load * self.ground_factor
            self.sys_load[i] = self.base_load + grounded_load
            self.sys_load[i] = self.sys_load[i-1] + self.dt * (self.sys_load[i] - self.sys_load[i-1]) / self.damping
            self.sys_load[i] = np.clip(self.sys_load[i], 0, self.max_load)
            # Thermal model
            power = min(self.sys_load[i] * self.heat_coeff, 1e6)
            self.temp[i] = self.temp[i-1] + self.dt * (
                (power - self.dissipation_coeff * (self.temp[i-1] - self.ambient_temp)) / self.thermal_mass
            )
            self.temp[i] = np.clip(self.temp[i], self.ambient_temp, self.max_temp)
            # Energy calculation
            self.energy += power * self.dt
            self.energy_vals[i] = self.energy
            # Efficiency: grounded load / input load
            self.efficiency_vals[i] = grounded_load / self.input_load[i] if self.input_load[i] > 0 else 0
            # Entropy calculation
            if i >= self.window_size:
                window = self.sys_load[i-self.window_size:i]
                hist, _ = np.histogram(window, bins=self.n_bins, range=(0, self.max_load), density=True)
                hist = hist + 1e-10  # Avoid log(0)
                self.entropy_vals[i] = entropy(hist, base=2)
        # Update plot data
        self.input_load_data = list(self.input_load)
        self.sys_load_data = list(self.sys_load)
        self.temp_data = list(self.temp)
        self.entropy_data = list(self.entropy_vals)
        self.energy_data = list(self.energy_vals)
        self.efficiency_data = list(self.efficiency_vals)
        self.max_frames = len(self.time)
        self.frame = 0
        # Save results for this cycle
        self.save_results()

    def save_results(self):
        try:
            # Append results for this cycle to the results_data list
            for i in range(len(self.time)):
                self.results_data.append({
                    'Cycle': self.cycle_count,
                    'Time': float(self.time[i]),  # Convert numpy float to Python float
                    'Input Load (tasks/s)': float(self.input_load[i]),
                    'System Load (tasks/s)': float(self.sys_load[i]),
                    'Temperature (°C)': float(self.temp[i]),
                    'Entropy (bits)': float(self.entropy_vals[i]),
                    'Energy (J)': float(self.energy_vals[i]),
                    'Efficiency': float(self.efficiency_vals[i])
                })
            # Write all results to the JSON file
            with open(self.results_file, 'w') as f:
                json.dump(self.results_data, f, indent=4)
            # Update status with summary
            self.status_text = (f"Cycle {self.cycle_count} | Max Temp: {max(self.temp_data):.1f}°C | "
                               f"Avg Entropy: {np.mean(self.entropy_vals):.2f} bits | "
                               f"Total Energy: {self.energy:.2f} J")
        except Exception as e:
            self.status_text = f"Error saving results to JSON: {str(e)}"

    def reset_simulation(self, instance):
        self.cycle_count = 0
        self.frame = 0
        self.energy = 0
        self.results_data = []  # Clear results data
        self.initialize_simulation()
        self.status_text = "Simulation Reset"

    def update(self, dt):
        self.frame = min(self.frame + 1, self.max_frames - 1)
        if self.frame >= self.max_frames - 1 and self.cycle_count < self.max_cycles:
            self.cycle_count += 1
            self.initialize_simulation()  # Restart simulation for next cycle
        self.label.text = f"[b]Status:[/b] [color=000000]{self.status_text}[/color]"
        self.plot_widget.canvas.before.clear()
        with self.plot_widget.canvas.before:
            # Draw background
            Color(1, 1, 1, 1)
            Rectangle(pos=self.plot_widget.pos, size=self.plot_widget.size)
            # Draw axes and labels
            Color(0, 0, 0, 1)
            plot_height = self.plot_widget.height / 5  # Increased to 5 plots
            for i in range(5):
                y = self.plot_widget.height - (i + 1) * plot_height
                Line(points=[50, y, self.plot_widget.width - 50, y], width=2)
                Line(points=[50, y - plot_height + 20, 50, y], width=2)
                # Add grid lines
                for j in range(1, 5):
                    y_grid = y - (j * (plot_height - 20) / 5)
                    Line(points=[50, y_grid, self.plot_widget.width - 50, y_grid], width=1, dash_length=5, dash_offset=5)
            # Plot scaling
            max_load_val = self.surge_load * 1.1
            max_T = max(self.temp_data) * 1.1 if max(self.temp_data) > self.initial_temp else self.initial_temp + 10
            max_entropy = max(self.entropy_data) * 1.1 if max(self.entropy_data) > 0 else np.log2(self.n_bins)
            max_energy = max(self.energy_data) * 1.1 if max(self.energy_data) > 0 else 100
            max_efficiency = max(self.efficiency_data) * 1.1 if max(self.efficiency_data) > 0 else 1
            # Draw plots
            for i in range(int(self.frame)):
                x = 50 + (i / self.max_frames) * (self.plot_widget.width - 100)
                # Input and System Load
                y_input = self.plot_widget.height - plot_height - (self.input_load_data[i] / max_load_val) * (plot_height - 20)
                y_sys = self.plot_widget.height - plot_height - (self.sys_load_data[i] / max_load_val) * (plot_height - 20)
                if i > 0:
                    Color(1, 0, 0, 1)  # Red for input
                    Line(points=[x_prev, y_input_prev, x, y_input], width=2)
                    Color(0, 0, 1, 1)  # Blue for system
                    Line(points=[x_prev, y_sys_prev, x, y_sys], width=2)
                x_prev, y_input_prev, y_sys_prev = x, y_input, y_sys
                # Temperature
                y_temp = self.plot_widget.height - 2 * plot_height - ((self.temp_data[i] - self.ambient_temp) / (max_T - self.ambient_temp)) * (plot_height - 20)
                if i > 0:
                    Color(0, 1, 0, 1)  # Green for temp
                    Line(points=[x_prev_t, y_temp_prev, x, y_temp], width=2)
                x_prev_t, y_temp_prev = x, y_temp
                # Entropy
                y_entropy = self.plot_widget.height - 3 * plot_height - (self.entropy_data[i] / max_entropy) * (plot_height - 20)
                if i > 0:
                    Color(0.5, 0, 0.5, 1)  # Purple for entropy
                    Line(points=[x_prev_e, y_entropy_prev, x, y_entropy], width=2)
                x_prev_e, y_entropy_prev = x, y_entropy
                # Energy
                y_energy = self.plot_widget.height - 4 * plot_height - (self.energy_data[i] / max_energy) * (plot_height - 20)
                if i > 0:
                    Color(1, 0.5, 0, 1)  # Orange for energy
                    Line(points=[x_prev_en, y_energy_prev, x, y_energy], width=2)
                x_prev_en, y_energy_prev = x, y_energy
                # Efficiency
                y_efficiency = self.plot_widget.height - 5 * plot_height - (self.efficiency_data[i] / max_efficiency) * (plot_height - 20)
                if i > 0:
                    Color(0, 0.5, 0.5, 1)  # Cyan for efficiency
                    Line(points=[x_prev_eff, y_efficiency_prev, x, y_efficiency], width=2)
                x_prev_eff, y_efficiency_prev = x, y_efficiency
            # Add labels
            Color(0, 0, 0, 1)
            self.plot_widget.canvas.after.clear()
            with self.plot_widget.canvas.after:
                Label(text="Load (tasks/s)", pos=(0, self.plot_widget.height - plot_height / 2), size=(50, 20))
                Label(text="Temp (°C)", pos=(0, self.plot_widget.height - 3 * plot_height / 2), size=(50, 20))
                Label(text="Entropy (bits)", pos=(0, self.plot_widget.height - 5 * plot_height / 2), size=(50, 20))
                Label(text="Energy (J)", pos=(0, self.plot_widget.height - 7 * plot_height / 2), size=(50, 20))
                Label(text="Efficiency", pos=(0, self.plot_widget.height - 9 * plot_height / 2), size=(50, 20))

class GroundingApp(App):
    def build(self):
        return GroundingWidget()

if __name__ == '__main__':
    GroundingApp().run()