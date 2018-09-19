#using Plots
#plotly()
using UnicodePlots

# constants
v_cheetah_max = 30;     # [m/s]
d_cheetah_min = 5;      # [m]
t_max = 5;              # [seconds]
t_step = 0.01;          # [seconds]
mass = 0.35;            # [kg]

# time
t = 0:t_step:t_max;

# x position over time
x = v_cheetah_max * t - (v_cheetah_max*t_max/2);  # [m]

# angle as a function of position
θ = atan.(x / d_cheetah_min);  # [rad]

# distance from camera to cheetah
d = d_cheetah_min ./ cos.(θ);  # [m]
println(lineplot(t, d, title="Distance from camera [m] vs time [s]"))

# angular velocity of camera
ω = v_cheetah_max ./ d;  # rad/s
println(lineplot(t, ω,
                 title="Angular velocity [rad/s] vs time [s]"))

# angular acceleration of camera
α = (ω[2:end] - ω[1:end-1])/t_step;  # rad/s²
println(lineplot(t[1:end-1], α,
                 title="Angular acceleration [rad/s²] vs time [s]"))

# torque (accelerating)
T = mass * α * 0.005;
println(lineplot(t[1:end-1], T,
                 title="Torque (accelerating) [Nm] vs time [s]"))

println("Torque (load) still needs to be calculated/added (maybe)?")

# display useful values
ω_max = round(maximum(ω) * 60/2π)
α_max = round(maximum(α))
T_max = round(maximum(T), 5)

# print em if not using the REPL
println()
println("Maximum angular velocity = $ω_max [rpm]")  # 6 rad/s
println("Maximum angular acceleration = $α_max [rad/s²]")  # 14 rad/s²
println("Maximum accelerating torque = $(T_max*100) [N⋅cm]")  # 0.02423 [Nm]
