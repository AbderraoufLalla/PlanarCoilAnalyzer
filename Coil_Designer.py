import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.constants import mu_0
from tkinter import font as tkfont

MU_0 = mu_0

def plot_planar_coil(x, y, outer_diameter, w, s, num_turns):
    inner_diameter = outer_diameter - 2 * (w + s) * num_turns
    if inner_diameter < 0:
        plt.close('all')
    else:
        plt.clf()
        plt.plot(x, y, color='blue', linewidth=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Customizing ticks on x-axis and y-axis with a step
        number_of_x_ticks = 10
        number_of_y_ticks = 10

        step_x = 1/1000
        step_y = step_x
        span = 10/1000


        x_ticks = np.arange(-span, span + step_x, step_x)
        y_ticks = np.arange(-span, span + step_y, step_y)

        plt.xticks(x_ticks, rotation=90)
        plt.yticks(y_ticks, rotation=0)

        plt.title(
            f'Planar Coil - R_out={outer_diameter / 2: .2f}mm - turns={num_turns: .2f} - w={w: .2f}mm - s={s: .2f}mm - R_in={inner_diameter / 2: .2f}mm',
            fontsize=6.5)
        plt.grid(True)

def generate_planar_coil(outer_diameter, w, s, rotation, num_turns):
    inner_diameter = outer_diameter - 2 * (w + s) * num_turns
    theta = np.linspace(0, 2 * np.pi * num_turns, 5000)
    radii = np.linspace(inner_diameter / 2, outer_diameter / 2, len(theta))
    x = radii * np.cos(rotation * theta)
    y = radii * np.sin(rotation * theta)
    return x, y

def generate_planar_coil_withLayers(outer_diameter, w, s, rotation, num_turns):
    inner_diameter = outer_diameter - 2 * (w + s) * num_turns
    theta = np.linspace(0, 2 * np.pi * num_turns, 10000)
    radii = np.linspace(inner_diameter / 2, outer_diameter / 2, len(theta))
    x = radii * np.cos(rotation * theta)
    y = radii * np.sin(rotation * theta)
    coil_points = concatenate_transpose(x, y)
    return coil_points

def plot_coil_3d(coil_points, observation_point):
    fig = plt.figure(100, figsize=(8, 6))
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    # Plot coil
    ax.plot(coil_points[:, 0], coil_points[:, 1], zs=0, zdir='z', linewidth=2, color='blue')

    # Plot observation point
    ax.scatter(observation_point[0], observation_point[1], observation_point[2], color='green', s=10, label='Observation Point')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Coil and Magnetic Field')
    ax.legend()
    #plt.show()

def concatenate_transpose(vector_row1, vector_row2):
    max_length = max(len(vector_row1), len(vector_row2))

    # Pad the vectors with zeros to match the maximum length
    vector_row1_padded = np.pad(vector_row1, (0, max_length - len(vector_row1)), mode='constant')
    vector_row2_padded = np.pad(vector_row2, (0, max_length - len(vector_row2)), mode='constant')

    # Transpose the row vectors to column vectors
    vector_column1 = vector_row1_padded.reshape(-1, 1)
    vector_column2 = vector_row2_padded.reshape(-1, 1)
    vector_column3 = np.zeros((max_length, 1), dtype=vector_row1.dtype)  # Create zero vector

    # Concatenate the column vectors
    concatenated_vector_transposed = np.concatenate((vector_column1, vector_column2, vector_column3), axis=1)

    return concatenated_vector_transposed

def fill_ratio(d_out, d_in):
    fill = (d_out - d_in)/(d_out + d_in)
    return fill

def average(a, b):
    average = 0.5 * (a + b)
    return average

def distance3D(point1, point2):
    p1 = np.array(point1, dtype=np.float64)
    p2 = np.array(point2, dtype=np.float64)
    distance = np.linalg.norm(p2 - p1)
    return distance

def calculate_inductance_CurrentSheet(d_out, w, s, N):
    # Permeability of free space
    mu0 = 4 * np.pi * 1e-7  # Permeability of free space
    d_in = d_out - 2 * (w + s) * N
    d_avg = average(d_out, d_in)
    p = fill_ratio(d_out, d_in)
    c = [1, 2.46, 0 , 0.2]
    L = 0.5 * mu0 * N**2 * d_avg * c[0] * ( math.log(c[1]/p) + c[2] * p + c[3] * pow(p, 2))
    return L

def biot_savart(coil_points, current, observation_point):
    B = np.array([0.0, 0.0, 0.0])  # Initialize magnetic field

    for i in range(len(coil_points)-1):
        dl = coil_points[i+1] - coil_points[i]  # Current element
        r = observation_point - coil_points[i]  # Vector from current element to observation point
        r_mag = np.linalg.norm(r)
        dl_cross_r = np.cross(dl, r)
        B += (mu_0 / (4 * np.pi)) * (current * dl_cross_r) / (r_mag**3)
    return B

def calculate_resonant_frequency(L, C_sensor, C_board, C_LCOM ):
    return 1 / (2 * math.pi * math.sqrt(L * (C_sensor+C_board+C_LCOM)))

def calculate_capacitance(L, f, C_board, C_LCOM):
    c = 1 / (L * pow(2 * math.pi * f, 2))
    c = c - C_board - C_LCOM
    return c

def capacitance_scout(f, L, d, w, s, num_turns):
    min = f / 10  # KHz
    # Capacitance test
    frequency_scout = np.arange(min, f + 1, f / 100)
    capacitance_scout = []
    for i in frequency_scout:
        capacitance_scout.append(calculate_capacitance(L, i))

    plt.figure(1, figsize=(8,6))
    plt.clf()
    plt.plot(frequency_scout, capacitance_scout)
    x = np.arange(min, max(frequency_scout) + 1, max(frequency_scout) / 20)
    y = np.arange(min, max(frequency_scout) + 1, max(frequency_scout) / 20)
    # y = np.arange(0, max(capacitance_scout) + 1, max(capacitance_scout)/10)
    plt.xticks(x)
    plt.yticks(y)
    plt.xlabel('Frequency(kHz)')
    plt.ylabel('Capacitance(F)')
    plt.title('Frequency/Capacitance')
    plt.grid()

def plot_coil_3d_multilayers(coil_points, layers, layer_spacing):
    fig = plt.figure(100, figsize=(8, 6))
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    set = np.arange(0, layers, 1)
    for i in set:
        # Apply the z translation to the coil
        translated_coil = coil_points.copy()
        translated_coil[:, 2] += layer_spacing * i

        # Plot the translated coil
        ax.plot(
            translated_coil[:, 0], translated_coil[:, 1], translated_coil[:, 2],
            linewidth=1, color='blue', label=f'Coil at z={layer_spacing}'
        )

    # Label axes and set plot title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ticks = np.arange(0, layers*layer_spacing*4, layers*layer_spacing*3/10)
    ax.set_zticks(ticks, [f'{tick:.3f}' for tick in ticks])
    ax.grid()
    ax.set_title('Coil and Magnetic Field')

def calculate_pista(current, thickness, Trise):
    ki = 0.024
    ke = 0.048

    bi = 0.44
    be = 0.44

    ci = 0.725
    ce = 0.725

    xi = ki * pow(Trise, bi)
    xe = ke * pow(Trise, be)

    Ai = pow(current/xi, 1/ci)
    Ae = pow(current / xe, 1 / ce)


    Wi = (Ai / (thickness * 1.378)) * 0.0254 # mils to mm
    We = (Ae / (thickness * 1.378)) * 0.0254 # mils to mm

    return float(Wi), float(We)

def calculate_distance(point1, point2):
    distance = abs(point2 - point1)
    return distance

def feasibility_test(d, w, s):
    for i in range(1, 100, 1):  # To calculate the feasible number of turns
        inner_diameter_pseudo = d - 2 * (w + s) * i
        count = i - 1
        if inner_diameter_pseudo < 0:
            break
    return count

def select_width(real_width, list_widths):
    distances = []
    for standard_width in list_widths:
        distance = calculate_distance(real_width, standard_width)
        distances.append(distance)
    index_selected_width = distances.index(min(distances))
    real_distance = real_width - list_widths[index_selected_width]
    if real_distance > 0:
        index_selected_width = index_selected_width + 1
    width = list_widths[index_selected_width]
    return width

def mm_to_mils(valore):
    mils = valore * 39.3701
    return mils

def mils_to_mm(valore):
    mm = valore * 0.0254
    return mm

def oz_to_meters(oz_per_sq_ft):
    grams_per_ounce = 28.3495
    sq_ft_to_sq_cm = 929.0304
    copper_density_g_per_cm3 = 8.96
    cm_to_micrometers = 10000
    g_per_sq_cm = (oz_per_sq_ft * grams_per_ounce) / sq_ft_to_sq_cm
    thickness_cm = g_per_sq_cm / copper_density_g_per_cm3
    thickness_meters = thickness_cm * cm_to_micrometers * 1e-6
    return thickness_meters

def total_inductance(L, num_turns, num_coils, layer_spacing):
    layer_spacing_mm = layer_spacing * 1000
    mutual_inductance_matrix = np.zeros((num_coils, num_coils))
    T_scaling = 1.5625 * num_turns ** 2 / (1.67 * num_turns ** 2 - 5.84 * num_turns + 65)
    for i in range(1,num_coils+1,1):
        for j in range(1, num_coils+1, 1):
            if i != j:
                distance = layer_spacing_mm * abs(j - i)
                M = 1 / (0.184 * ((distance) ** 3) - (0.525 * distance ** 2) + (1.038 * distance) + 1.001)
                mutual_inductance_matrix[i-1,j-1] = M
    L_mutual_sum = sum_matrix(mutual_inductance_matrix)
    scale_factor = L_mutual_sum * T_scaling + num_coils
    L_total = scale_factor * L
    return float(L_total)

def skin_depth(frequency, copper_thickness):
    resistivity_copper = 1.68e-8
    permeability = 4 * math.pi * 10 ** -7
    omega = 2 * math.pi * frequency
    delta = math.sqrt(2 * resistivity_copper / (omega * permeability))
    skin_depths = copper_thickness / delta
    return delta, skin_depths

def sum_matrix(matrix):
    total_sum = 0
    for row in matrix:
        for element in row:
            total_sum += element
    return total_sum

# Magnetic permeability of copper
mu_copper = 4 * np.pi * pow(10, -4) # H/m

#List of stadard widths
a = list(np.arange(3, 13, 1)) # Fine pitch traces in mils
b = [15, 20, 25, 30] # Standard pitch traces in mils
c = [50, 75, 100, 150]  # Standard pitch traces in mils

# Board Constants
C_board = 12 * 1e-12
C_LCOM = 12 * 1e-12

standard_widths_mils = a + b + c
standard_widths=[]
for i in standard_widths_mils: # Transform whole list to milimeters
    number = mils_to_mm(i)
    number = round(number, 3)
    standard_widths.append(number)

def Construct():
    global d, s, w, rotation, num_turns, f, layers, layer_spacing, current, thickness, trise, ambient_temp, standard_We, standard_Wi, L, We, Wi, coil_name
    d = float(d_entry.get())
    s = float(spacing_entry.get())
    w = float(width_entry.get())
    rotation = int(r_entry.get())
    num_turns = float(n_entry.get())
    f = float(Frequency_entry.get())
    layers = int(Layers_entry.get())
    num_coils = layers
    layer_spacing = float(LayerSpacing_entry.get())
    current = float(Current_entry.get())
    thickness = float(thickness_entry.get())
    trise = float(Trise_entry.get())
    ambient_temp = 25
    coil_name = coilname_entry.get()

    d = d / 1000
    s = s /1000
    w = w / 1000
    layer_spacing = layer_spacing / 1000
    f = f * 1000
    copper_thickness = oz_to_meters(thickness)

    #Board Constants
    C_board = 12 * 1e-12
    C_LCOM = 12 * 1e-12

    # Analytic coil
    inner_diameter = d - 2 * (w + s) * num_turns
    x, y = generate_planar_coil(d, w, s, rotation, num_turns)
    L = calculate_inductance_CurrentSheet(d, w, s, num_turns)
    L_total = total_inductance(L, num_turns, num_coils, layer_spacing)
    Capacitance = calculate_capacitance(L_total,f, C_board, C_LCOM)
    Wi, We = calculate_pista(current, thickness, trise)
    standard_We = select_width(We, standard_widths)
    standard_Wi = select_width(Wi, standard_widths)

    #Standard coil
    inner_diameter_standard = d - 2 * (We + s) * num_turns
    x_standard, y_standard = generate_planar_coil(d, standard_We/1000, s, rotation, num_turns) # Transfer to meters
    L_standard = calculate_inductance_CurrentSheet(d, standard_We/1000, s, num_turns) # Transfer to meters
    L_total_standard = total_inductance(L_standard, num_turns, num_coils, layer_spacing)
    Capacitance_standard = calculate_capacitance(L_total_standard, f, C_board, C_LCOM)
    skin_depth_value, skin_depths = skin_depth(f, copper_thickness)
    Sensor_frequency = calculate_resonant_frequency(L_total_standard, Capacitance_standard, C_board, C_LCOM)

    if inner_diameter < 0: # THE FEASIBILITY OF THE GEOMETRY
        message1.config(text="Invalid coil design: Inner diameter is non-positive, feasible number of turns is not respected. Adjust parameters.", fg="red", font=("Arial", 12, "bold"))
        message_widgets = [
            message2, message3, message4, message5, message6, message7, message8,
            message9, message10, message11, message12, message13, message14,
            message15, message16, message17, message18, message19, message20, message21, message22, message23
        ]
        for widget in message_widgets:
            widget.config(text="")
        plt.close('all')

    else:
        if rotation == 1 or rotation == 2:
            d_avg = average(d, inner_diameter)
            p = fill_ratio(d, inner_diameter)

            # Coil points
            coil_points = concatenate_transpose(x, y)
            feasibility = feasibility_test(d, w, s)
            feasibility_standard = feasibility_test(d, standard_We/1000, s)

            # Observation point
            observation_point = np.array([0.0, 0.0, 0.0])  # Example observation point in millimeters
            observation_point1 = np.array([0.0, 0.0, 1.0])
            observation_point2 = np.array([0.0, 0.0, 1.3])
            observation_point3 = np.array([0.0, 0.0, 1.7])
            observation_point4 = np.array([0.0, 0.0, 2.0])

            # Calculate magnetic field
            B_field = biot_savart(coil_points, current, observation_point) * 1000
            B_field1 = biot_savart(coil_points, current, observation_point1) * 1000
            B_field2 = biot_savart(coil_points, current, observation_point2) * 1000
            B_field3 = biot_savart(coil_points, current, observation_point3) * 1000
            B_field4 = biot_savart(coil_points, current, observation_point4) * 1000

            message1.config(text=f'COIL GEOMETRY AND DATA - NON STANDARD', fg="red", font=font)
            message2.config(text=f'Inner diameter = {inner_diameter*1000:.3} mm/{(mm_to_mils(inner_diameter*1000)):.3}mils | Fill ratio = {(inner_diameter/d): .2f}', font=font)

            message3.config(text=f"Self inductance per layer = {L * 10 ** 6:.4} uH | Coil inductance = {L_total*1e6:.4} uH", fg="black", font=font)
            message4.config(text=f'LC capacitance = {(round(Capacitance,12) * 1e12):.9} pF', font=font)
            message5.config(text=f'Feasible number of turns = {feasibility}', font=font)
            message6.config(text=f'Sensor frequency = {f*1e-3} KHz', font=font)


            message7.config(text=f'MAGNETIC DATA', fg="red", font=font)
            message8.config(text=f'Magnetic field at {observation_point}mm = {B_field} mT', font=font)
            message9.config(text=f'Magnetic field at {observation_point1}mm = {B_field1} mT', font=font)
            message10.config(text=f'Magnetic field at {observation_point2}mm = {B_field2} mT', font=font)
            message11.config(text=f'Magnetic field at {observation_point3}mm = {B_field3} mT', font=font)
            message12.config(text=f'Magnetic field at {observation_point4}mm = {B_field4} mT', font=font)

            message13.config(text=f'PCB LAYOUT', fg="red", font=font)
            message14.config(text=f'Wire width - External layers: {We:.3} mm/{(mm_to_mils(We)):.3}mils', font=font)
            message15.config(text=f'Wire width - Internal layers: {Wi:.3} mm/{(mm_to_mils(Wi)):.3}mils', font=font)

            message16.config(text=f'PCB LAYOUT - STANDARD | Ambient temperature: {ambient_temp} degrees celsius', fg="red", font=font)
            message17.config(text=f'Standard PCB wire widths: {standard_widths}mm', font=font)
            message18.config(text=f'Skin depth at {f*1e-6:.4} MHz for copper is {skin_depth_value * 1e6:.2f} micrometers | Number of skin depths is: {skin_depths:.2} | Advised copper thickness less than: {0.5*skin_depth_value*1e6:.4} micrometers', font=font)
            message19.config(text=f'Wire widths: External layers: {standard_We:.3} mm/{(mm_to_mils(standard_We)):.3}mils | Internal layers: {standard_Wi:.3} mm/{(mm_to_mils(standard_Wi)):.3}mils', font=font)
            message20.config(text=f'Standard self inductance per layer: {(L_standard * pow(10, 6)):.4} uH | Standard coil inductance: {(L_total_standard * pow(10, 6)):.4} uH', font=font)
            message21.config(text=f'Standard LC capacitance: {round(Capacitance_standard,12)*1e12:.9} pF', font=font)
            message22.config(text=f'Feasible standard number of turns = {feasibility_standard}', font=font)
            message23.config(text=f'Sensor frequency = {Sensor_frequency*1e-3:.5} KHz', font=font)
        else:
            message1.config(text="Invalid coil design: Invalid rotation.",fg="red", font=("Arial", 12, "bold"))
            message_widgets = [
                message2, message3, message4, message5, message6, message7, message8,
                message9, message10, message11, message12, message13, message14,
                message15, message16, message17, message18, message19, message20, message21, message22, message23
            ]
            for widget in message_widgets:
                widget.config(text="")
            plt.close('all')

def Plot_2D(): # WINDOW 1: 2D PLOT COIL
    fig = plt.figure(1, figsize=(8, 6)) # WINDOW 1: 2D PLOT COIL
    inner_diameter = d - 2 * (w + s) * num_turns
    x, y = generate_planar_coil(d, w, s, rotation, num_turns)
    plt.clf()
    plt.plot(x, y, color='blue', linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')

    step_x = 1 / 1000
    step_y = step_x
    span = 5 /1000

    x_ticks = np.arange(-span, span + step_x, step_x)
    y_ticks = np.arange(-span, span + step_y, step_y)

    plt.xticks(x_ticks, rotation=90)
    plt.yticks(y_ticks, rotation=0)
    plt.grid(True)
    plt.get_current_fig_manager().set_window_title("2D coil plot")
    plt.show()

def Plot_3D(): # WINDOW 1: 2D PLOT COIL
    coil_points = generate_planar_coil_withLayers(d, w, s, rotation, num_turns)
    fig = plt.figure(2, figsize=(8, 6))
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    set = np.arange(0, layers, 1)
    for i in set:
        # Apply the z translation to the coil
        translated_coil = coil_points.copy()
        translated_coil[:, 2] += layer_spacing * i

        # Plot the translated coil
        ax.plot(
            translated_coil[:, 0], translated_coil[:, 1], translated_coil[:, 2],
            linewidth=1, color='blue', label=f'Coil at z={layer_spacing}'
        )
    plt.get_current_fig_manager().set_window_title("3D coil plot")
    step_x = 1 / 1000
    step_y = step_x
    span = 5 /1000

    x_ticks = np.arange(-span, span + step_x, step_x)
    y_ticks = x_ticks
    z_ticks = np.arange(0, layer_spacing * 3 * layers, layer_spacing)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    plt.show()

def Capacitance():
    #Board Constants
    C_board = 12 * 1e-12
    C_LCOM = 12 * 1e-12
    fig = plt.figure(3, figsize=(8, 6))
    plt.clf()
    min = f / 10  # KHz
    # Capacitance test
    frequency_scout = np.arange(min, f + 1, f / 100)
    capacitance_scout = []
    for i in frequency_scout:
        capacitance_scout.append(calculate_capacitance(L, i, C_board, C_LCOM))

    plt.plot(frequency_scout, capacitance_scout)
    x = np.arange(min, max(frequency_scout) + 1, max(frequency_scout) / 20)
    # y = np.arange(0, max(capacitance_scout) + 1, max(capacitance_scout)/10)
    plt.xticks(x)
    # plt.yticks(y)
    plt.xlabel('Frequency(kHz)')
    plt.ylabel('Capacitance(F)')
    plt.title('Frequency/Capacitance')
    plt.grid()
    plt.get_current_fig_manager().set_window_title("Capacitance selection")
    plt.show()

def InductanceVSLayers():
    num_coils = 8
    L_total = []
    x = []
    for i in range(1, num_coils + 1, 1):
        all = total_inductance(L, num_turns, i, layer_spacing) * 1e6
        round(all, 9)
        L_total.append(all)
        x.append(i)

    plt.plot(x, L_total, marker='o', linestyle='-', color='b', label='Inductance')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Number of layers')
    plt.ylabel('Inductance (uH)')
    plt.title('Inductance vs Number of Layers')
    plt.legend()

    # Annotate each point with its value
    for i, txt in enumerate(L_total):
        plt.annotate(f'{txt:.3f}', (x[i], L_total[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.show()

def Write_Coil_Data():
    num_coils = layers
    copper_thickness = oz_to_meters(thickness)

    inner_diameter = d - 2 * (w + s) * num_turns
    d_avg = average(d, inner_diameter)
    p = fill_ratio(d, inner_diameter)

    # Analytic coil
    x, y = generate_planar_coil(d, w, s, rotation, num_turns)
    L = calculate_inductance_CurrentSheet(d, w, s, num_turns)
    L_total = total_inductance(L, num_turns, num_coils, layer_spacing)
    Capacitance = calculate_capacitance(L_total, f, C_board, C_LCOM)
    Wi, We = calculate_pista(current, thickness, trise)
    standard_We = select_width(We, standard_widths)
    standard_Wi = select_width(Wi, standard_widths)
    feasibility = feasibility_test(d, w, s)

    # Standard coil
    inner_diameter_standard = d - 2 * (We + s) * num_turns
    x_standard, y_standard = generate_planar_coil(d, standard_We / 1000, s, rotation, num_turns)  # Transfer to meters
    L_standard = calculate_inductance_CurrentSheet(d, standard_We / 1000, s, num_turns)  # Transfer to meters
    L_total_standard = total_inductance(L_standard, num_turns, num_coils, layer_spacing)
    Capacitance_standard = calculate_capacitance(L_total_standard, f, C_board, C_LCOM)
    skin_depth_value, skin_depths = skin_depth(f, copper_thickness)
    Sensor_frequency = calculate_resonant_frequency(L_total_standard, Capacitance_standard, C_board, C_LCOM)
    feasibility_standard = feasibility_test(d, standard_We / 1000, s)

    # Coil points
    coil_points = concatenate_transpose(x, y)
    # Observation point
    observation_point = np.array([0.0, 0.0, 0.0])  # Example observation point in millimeters
    observation_point1 = np.array([0.0, 0.0, 1.0])
    observation_point2 = np.array([0.0, 0.0, 1.3])
    observation_point3 = np.array([0.0, 0.0, 1.7])
    observation_point4 = np.array([0.0, 0.0, 2.0])
    # Calculate magnetic field
    B_field = biot_savart(coil_points, current, observation_point) * 1000
    B_field1 = biot_savart(coil_points, current, observation_point1) * 1000
    B_field2 = biot_savart(coil_points, current, observation_point2) * 1000
    B_field3 = biot_savart(coil_points, current, observation_point3) * 1000
    B_field4 = biot_savart(coil_points, current, observation_point4) * 1000

    messages = [
        f'COIL GEOMETRY AND DATA - NON STANDARD',
        f'Diameter= {d*1e3:.3}mm / {(mm_to_mils(d*1e3)):.5} mils',
        f'Wire spacing= {s*1e3:.3}mm / {(mm_to_mils(s*1e3)):.5} mils',
        f'Wire width= {w*1e3:.3}mm / {(mm_to_mils(w*1e3)):.5} mils',
        f'Wire thickness= {thickness} oz-copper / {(oz_to_meters(thickness))*1e6:.3} micrometers',
        f'Rotation= {rotation} (1: Counter-clk; 2: clk)',
        f'Number of turns= {num_turns}',
        f'Number of layers (coils)= {layers}',
        f'Layer spacing= {layer_spacing*1e3}mm / {(mm_to_mils(layer_spacing*1e3)):.5} mils',
        f'Temperature rise= {trise} degrees celsius',
        f'Current= {current} Amps',
        f'--------------------------------------------------------------------------------------------------------------',
        f'Inner diameter = {inner_diameter * 1000:.3} mm/{(mm_to_mils(inner_diameter * 1000)):.3}mils | Fill ratio = {(inner_diameter / d): .2f}',
        f"Self inductance per layer = {L * 10 ** 6:.4} uH | Coil inductance = {L_total * 1e6:.4} uH",
        f'LC capacitance = {(round(Capacitance, 12) * 1e12):.9} pF',
        f'Feasible number of turns = {feasibility}',
        f'Sensor frequency = {f * 1e-3} KHz',
        f'--------------------------------------------------------------------------------------------------------------',
        f'MAGNETIC DATA',
        f'Magnetic field at {observation_point}mm = {B_field} mT',
        f'Magnetic field at {observation_point1}mm = {B_field1} mT',
        f'Magnetic field at {observation_point2}mm = {B_field2} mT',
        f'Magnetic field at {observation_point3}mm = {B_field3} mT',
        f'Magnetic field at {observation_point4}mm = {B_field4} mT',
        f'--------------------------------------------------------------------------------------------------------------',
        f'PCB LAYOUT',
        f'Wire width - External layers: {We:.3} mm/{(mm_to_mils(We)):.3}mils',
        f'Wire width - Internal layers: {Wi:.3} mm/{(mm_to_mils(Wi)):.3}mils',
        f'--------------------------------------------------------------------------------------------------------------',
        f'PCB LAYOUT - STANDARD | Ambient temperature: {ambient_temp} degrees celsius',
        f'Standard PCB wire widths: {standard_widths}mm',
        f'Skin depth at {f * 1e-6:.4} MHz for copper is {skin_depth_value * 1e6:.2f} micrometers | Number of skin depths is: {skin_depths:.2} | Advised copper thickness less than: {0.5 * skin_depth_value * 1e6:.4} micrometers',
        f'Wire widths: External layers: {standard_We:.3} mm/{(mm_to_mils(standard_We)):.3}mils | Internal layers: {standard_Wi:.3} mm/{(mm_to_mils(standard_Wi)):.3}mils',
        f'Standard self inductance per layer: {(L_standard * pow(10, 6)):.4} uH | Standard coil inductance: {(L_total_standard * pow(10, 6)):.4} uH',
        f'Standard LC capacitance: {round(Capacitance_standard, 12) * 1e12:.9} pF',
        f'Feasible standard number of turns = {feasibility_standard}',
        f'Sensor frequency = {Sensor_frequency * 1e-3:.5} KHz'
    ]
    with open(f'{coil_name}.txt', 'w') as file:
        for message in messages:
            file.write(message + '\n')




# Create the main application window
app = tk.Tk()
app.title("Planar Coil Designer")

# Set the initial size of the window and make it resizable
app.geometry("700x750")  # Width x Height
app.minsize(700, 750)  # Minimum size to maintain the layout structure

# Define custom fonts for buttons and labels
bold_font = tkfont.Font(family="Garamond", size=11, weight="bold")

# Define colors for the interface
bg_color = "white"  # Background color for frames
button_color = "white"  # Background color for buttons
button_hover_color = "yellow"  # Button hover effect color
text_color = "black"  # Text color for buttons

# Set background color for the application window
app.configure(bg=bg_color)

# Create a canvas for scrolling
canvas = tk.Canvas(app, bg=bg_color)
scrollbar = tk.Scrollbar(app, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg=bg_color)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Pack canvas and scrollbar
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Create button frame
button_frame = tk.Frame(scrollable_frame, bg=bg_color)
button_frame.pack(fill=tk.BOTH, padx=0, pady=0)

button1 = tk.Button(
    button_frame,
    text="Construct Coil",
    font=bold_font,
    bg=button_color,
    fg=text_color,
    command=Construct,
)
button1.pack(fill=tk.X, pady=0)

button2 = tk.Button(
    button_frame,
    text="Plot Coil 2D",
    font=bold_font,
    bg=button_color,
    fg=text_color,
    command=Plot_2D,
)
button2.pack(fill=tk.X, pady=0)

button3 = tk.Button(
    button_frame,
    text="Plot Coil 3D",
    font=bold_font,
    bg=button_color,
    fg=text_color,
    command=Plot_3D,
)
button3.pack(fill=tk.X, pady=0)

button4 = tk.Button(
    button_frame,
    text="Capacitance selection",
    font=bold_font,
    bg=button_color,
    fg=text_color,
    command=Capacitance,
)
button4.pack(fill=tk.X, pady=0)

button5 = tk.Button(
    button_frame,
    text="Inductance Vs Layers",
    font=bold_font,
    bg=button_color,
    fg=text_color,
    command=InductanceVSLayers,
)
button5.pack(fill=tk.X, pady=0)

button6 = tk.Button(
    button_frame,
    text="Export coil data",
    font=bold_font,
    bg=button_color,
    fg=text_color,
    command=Write_Coil_Data,
)
button6.pack(fill=tk.X, pady=0)


def on_enter(event):
    event.widget.configure(bg=button_hover_color)

def on_leave(event):
    event.widget.configure(bg=button_color)

# Add hover effects to buttons
for button in [button1, button2, button3, button4, button5, button6]:
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)

# Create and add widgets
frame_width = 10
pad = 0
font = ("Georgia", 10,)
pos = 'w'
side = 'left'

d_frame = tk.Frame(scrollable_frame, width=frame_width, bg=bg_color)
d_frame.pack(anchor=pos)
d_label = tk.Label(d_frame, text="d(mm)=", font=font, bg=bg_color)
d_label.pack(side=side, padx=pad)
d_entry = tk.Entry(d_frame, width=frame_width, font=font)
d_entry.pack(side='left', padx=pad)

spacing_frame = tk.Frame(scrollable_frame, width=frame_width, bg=bg_color)
spacing_frame.pack(anchor=pos)
spacing_label = tk.Label(spacing_frame, text="s(mm)=", font=font, bg=bg_color)
spacing_label.pack(side='left', padx=pad)
spacing_entry = tk.Entry(spacing_frame, width=frame_width, font=font)
spacing_entry.pack(side='left', padx=pad, expand=True, fill='x')

width_frame = tk.Frame(scrollable_frame, width=frame_width, bg=bg_color)
width_frame.pack(anchor=pos)
width_label = tk.Label(width_frame, text="w(mm)=", font=font, bg=bg_color)
width_label.pack(side='left', padx=pad)
width_entry = tk.Entry(width_frame, width=frame_width, font=font)
width_entry.pack(side='left', padx=pad)

r_frame = tk.Frame(scrollable_frame, width=frame_width, bg=bg_color)
r_frame.pack(anchor=pos)
r_label = tk.Label(r_frame, text="Rotation (1: Counter-clk; 2: clk)=", font=font, bg=bg_color)
r_label.pack(side='left', padx=pad)
r_entry = tk.Entry(r_frame, width=20, font=font)
r_entry.pack(side='left', padx=pad)

n_frame = tk.Frame(scrollable_frame, width=frame_width, bg=bg_color)
n_frame.pack(anchor=pos)
n_label = tk.Label(n_frame, text="Number of turns=", font=font, bg=bg_color)
n_label.pack(side='left', padx=pad)
n_entry = tk.Entry(n_frame, width=20, font=font)
n_entry.pack(side='left', padx=pad)

Frequency_frame = tk.Frame(scrollable_frame, width=frame_width, bg=bg_color)
Frequency_frame.pack(anchor=pos)
Frequency_label = tk.Label(Frequency_frame, text="Frequency(KHz)=", font=font, bg=bg_color)
Frequency_label.pack(side='left', padx=pad)
Frequency_entry = tk.Entry(Frequency_frame, width=20, font=font)
Frequency_entry.pack(side='left', padx=pad)

Current_frame = tk.Frame(scrollable_frame, width=frame_width, bg=bg_color)
Current_frame.pack(anchor=pos)
Current_label = tk.Label(Current_frame, text="Current(amps)=", font=font, bg=bg_color)
Current_label.pack(side='left', padx=pad)
Current_entry = tk.Entry(Current_frame, width=20, font=font)
Current_entry.pack(side='left', padx=pad)

Layers_frame = tk.Frame(scrollable_frame, width=frame_width, bg=bg_color)
Layers_frame.pack(anchor=pos)
Layers_label = tk.Label(Layers_frame, text="Number of layers=", font=font, bg=bg_color)
Layers_label.pack(side='left', padx=pad)
Layers_entry = tk.Entry(Layers_frame, width=20, font=font)
Layers_entry.pack(side='left', padx=pad)

LayerSpacing_frame = tk.Frame(scrollable_frame, width=frame_width, bg=bg_color)
LayerSpacing_frame.pack(anchor=pos)
LayerSpacing_label = tk.Label(LayerSpacing_frame, text="Layers spacing=", font=font, bg=bg_color)
LayerSpacing_label.pack(side='left', padx=pad)
LayerSpacing_entry = tk.Entry(LayerSpacing_frame, width=20, font=font)
LayerSpacing_entry.pack(side='left', padx=pad)

thickness_frame = tk.Frame(scrollable_frame, width=frame_width, bg=bg_color)
thickness_frame.pack(anchor=pos)
thickness_label = tk.Label(thickness_frame, text="Wire thickness(oz)=", font=font, bg=bg_color)
thickness_label.pack(side='left', padx=pad)
thickness_entry = tk.Entry(thickness_frame, width=20, font=font)
thickness_entry.pack(side='left', padx=pad)

Trise_frame = tk.Frame(scrollable_frame, width=frame_width, bg=bg_color)
Trise_frame.pack(anchor=pos)
Trise_label = tk.Label(Trise_frame, text="Temperature rise(celsius)=", font=font, bg=bg_color)
Trise_label.pack(side='left', padx=pad)
Trise_entry = tk.Entry(Trise_frame, width=10, font=font)
Trise_entry.pack(side='right', padx=pad, ipadx=10)

coilname_frame = tk.Frame(scrollable_frame, width=frame_width, bg=bg_color)
coilname_frame.pack(anchor=pos)
coilname_label = tk.Label(coilname_frame, text="Coil name :", font=font, bg=bg_color)
coilname_label.pack(side='left', padx=pad)
coilname_entry = tk.Entry(coilname_frame, width=10, font=font)
coilname_entry.pack(side='right', padx=pad, ipadx=10)


# Add messages
for i in range(1, 24):
    globals()[f"message{i}"] = tk.Label(scrollable_frame, text="", bg=bg_color)
    globals()[f"message{i}"].pack(anchor='w')

# Start the Tkinter event loop
app.mainloop()