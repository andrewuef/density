import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import colorsys
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import random

# Load data (assuming this is done once when the app starts)
data = pd.read_excel("DensityData.xlsx")
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Function to create a more subtle color gradient
def create_color_gradient(hex_color, n):
    rgb = tuple(int(hex_color[1:][i:i+2], 16) for i in (0, 2, 4))
    hsv = colorsys.rgb_to_hsv(*[x/255.0 for x in rgb])
    return [colorsys.hsv_to_rgb(
        hsv[0],
        max(0, hsv[1] - 0.1 * i),  # Reduce saturation more gradually
        min(1, hsv[2] + 0.1 * i)   # Increase brightness more gradually
    ) for i in range(n)]

# Function to calculate current volumes from manufacturing data
def calculate_volumes(masses, densities):
    return masses / densities

# Empirical relationship for machine ratio settings
def machine_ratio_setting(volume_change):
    return 1 + 0.1 * volume_change  # example relationship

# Optimization function updates
def monte_carlo_simulation(dimensions, num_simulations, density_center, density_variation, dimension_variations, candies_per_package):
    # Calculate the total number of individual candies
    total_candies = num_simulations * candies_per_package

    # Generate dimensions for all individual candies
    sampled_dimensions = np.random.normal(dimensions, dimension_variations, (total_candies, 3))
    
    # Calculate volumes for all candies
    volumes = np.prod(sampled_dimensions, axis=1)
    
    # Generate densities for all candies
    densities = np.random.normal(density_center, density_variation, total_candies)
    
    # Calculate masses for all candies
    individual_masses = volumes * densities

    # Reshape the results into packages
    package_dimensions = sampled_dimensions.reshape(num_simulations, candies_per_package, 3)
    package_volumes = volumes.reshape(num_simulations, candies_per_package)
    package_densities = densities.reshape(num_simulations, candies_per_package)
    package_masses = individual_masses.reshape(num_simulations, candies_per_package)

    # Calculate package dimensions
    # Assuming dimension 3 is height and candies are stacked
    package_length = np.max(package_dimensions[:, :, 0], axis=1)
    package_width = np.max(package_dimensions[:, :, 1], axis=1)
    package_height = np.sum(package_dimensions[:, :, 2], axis=1)
    
    # Calculate total package masses
    total_package_masses = np.sum(package_masses, axis=1)

    return sampled_dimensions, volumes, densities, individual_masses, package_dimensions, package_length, package_width, package_height, total_package_masses


def objective_function(dimensions, num_simulations, density_center, density_variation, dimension_variations, lower_spec, declared_mass, candies_per_package):
    global tested_dimensions
    tested_dimensions.append(dimensions)
    
    _, _, _, individual_masses,_,_,_,_,_ = monte_carlo_simulation(dimensions, num_simulations, density_center, density_variation, dimension_variations,candies_per_package)
    
    # Generate package masses more efficiently
    package_masses = np.sum(individual_masses.reshape(-1, candies_per_package), axis=1)
    
    mean_package_mass = np.mean(package_masses)
    std_dev_package_mass = np.std(package_masses)
    declared_package_mass = declared_mass * candies_per_package
    target_package_spec = declared_package_mass + 0.5 * std_dev_package_mass
    lower_package_spec = lower_spec * candies_per_package
    upper_package_spec = target_package_spec + (target_package_spec - lower_package_spec)
    
    mean_target_alignment = mean_package_mass - target_package_spec
    within_spec = np.sum((package_masses >= lower_package_spec) & (package_masses <= upper_package_spec)) / len(package_masses)
    cpm = (upper_package_spec - lower_package_spec) / (6 * np.sqrt(std_dev_package_mass**2 + (mean_package_mass - target_package_spec)**2))
    
    return (mean_target_alignment ** 2) * 1000 + (1 - within_spec) * 100 - cpm * 10


# Constraint function for optimization
def constraint_function(dimensions_to_optimize, fixed_dims, dims_to_optimize, num_simulations, density_center, density_variation, dimension_variations, lower_spec, declared_mass, candies_per_package):
    full_dimensions = []
    optimize_index = 0
    for i, fixed in enumerate(fixed_dims):
        if fixed is not None:
            full_dimensions.append(fixed)
        else:
            full_dimensions.append(dimensions_to_optimize[optimize_index])
            optimize_index += 1
    
    _, _, _, individual_masses,_,_,_,_,_ = monte_carlo_simulation(np.array(full_dimensions), num_simulations, density_center, density_variation, dimension_variations,candies_per_package)
    
    # Generate package masses
    package_masses = np.array([np.sum(np.random.choice(individual_masses, size=candies_per_package)) for _ in range(num_simulations)])
    
    mean_package_mass = np.mean(package_masses)
    std_dev_package_mass = np.std(package_masses)
    declared_package_mass = declared_mass * candies_per_package
    target_package_spec = declared_package_mass + 0.5 * std_dev_package_mass
    
    # The constraint ensures that the mean package mass is above the target specification
    return mean_package_mass - target_package_spec + 1e-4

# Function to perform optimization
def perform_optimization(dims_to_optimize, fixed_dims, dimension_min, dimension_max, num_simulations, density_center, density_variation, dimension_variations, lower_spec, declared_mass, candies_per_package):
    global tested_dimensions
    tested_dimensions = []

    # Prepare bounds and initial guess for optimization
    bounds = []
    initial_guess = []
    for i, optimize in enumerate(dims_to_optimize):
        if optimize:
            bounds.append((dimension_min[i], dimension_max[i]))
            initial_guess.append((dimension_min[i] + dimension_max[i]) / 2)

    def objective_function_wrapper(dimensions_to_optimize):
        full_dimensions = []
        optimize_index = 0
        for i, optimize in enumerate(dims_to_optimize):
            if optimize:
                full_dimensions.append(dimensions_to_optimize[optimize_index])
                optimize_index += 1
            else:
                full_dimensions.append(fixed_dims[i])
        
        return objective_function(np.array(full_dimensions), num_simulations, density_center, density_variation, 
                                  dimension_variations, lower_spec, declared_mass, candies_per_package)

    def constraint_function_wrapper(dimensions_to_optimize):
        full_dimensions = []
        optimize_index = 0
        for i, optimize in enumerate(dims_to_optimize):
            if optimize:
                full_dimensions.append(dimensions_to_optimize[optimize_index])
                optimize_index += 1
            else:
                full_dimensions.append(fixed_dims[i])
        
        return constraint_function(dimensions_to_optimize, fixed_dims, dims_to_optimize, num_simulations, 
                                   density_center, density_variation, dimension_variations, 
                                   lower_spec, declared_mass, candies_per_package)

    result = minimize(
        objective_function_wrapper,
        initial_guess,
        method="COBYLA",
        bounds=bounds,
        constraints=({'type': 'ineq', 'fun': constraint_function_wrapper})
    )

    return result, tested_dimensions

def create_optimization_plots(results, candies_per_package, sampled_dimensions, volumes, densities, individual_masses, package_length, package_width, package_height, total_package_masses):
    # Calculate package masses
    package_masses = np.sum(individual_masses.reshape(-1, candies_per_package), axis=1)

    mean_package_mass = np.mean(package_masses)
    std_dev_package_mass = np.std(package_masses)
    declared_package_mass = results['declared_mass'] * candies_per_package
    target_package_spec = declared_package_mass + 0.5 * std_dev_package_mass
    lower_package_spec = results['lower_spec'] * candies_per_package
    upper_package_spec = target_package_spec + (target_package_spec - lower_package_spec)

    fig = make_subplots(
            rows=2, cols=2, 
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "scatter3d"}]],
            subplot_titles=("Package Mass Distribution", "Dimensions", "Generated Dimension Box Plots", "Generated Mass vs. Density vs. Volume"),
            vertical_spacing=0.2
        )

    # Package Mass distribution
    fig.add_trace(go.Histogram(x=total_package_masses, name="Package Mass"), row=1, col=1)

    counts, bins = np.histogram(package_masses, bins='auto')
    y_max = max(counts)

    # Add vertical lines
    for line, color, name in [
        (target_package_spec, "green", "Target Spec"),
        (lower_package_spec, "red", "Lower Spec Limit"),
        (upper_package_spec, "red", "Upper Spec Limit"),
        (mean_package_mass, "orange", "Mean Mass")
    ]:
        fig.add_shape(
            type="line", x0=line, x1=line, y0=0, y1=y_max,
            line=dict(color=color, dash="dash"), xref='x1', yref='y1'
        )
        fig.add_trace(go.Scatter(
            x=[line], y=[0], mode='markers',
            marker=dict(size=0, color=color),
            hovertemplate=f"{name}: %{{x}}<extra></extra>",
            showlegend=False
        ), row=1, col=1)

    # Dimensions plot
    if 'tested_dimensions' in results:
        # Plot for optimization results
        tested_dimensions = np.array(results['tested_dimensions'])
        color_range = np.array(range(len(tested_dimensions)))

        fig.add_trace(go.Scatter(x=tested_dimensions[:, 0], y=tested_dimensions[:, 1], 
                                mode='markers', marker=dict(color=color_range, 
                                                            colorscale='Viridis', 
                                                            size=5), 
                                name="Tested Dimensions"), row=1, col=2)

        fig.add_trace(go.Scatter(x=[results['optimized_dimensions'][0]], 
                                y=[results['optimized_dimensions'][1]], 
                                mode='markers', marker_symbol='star', 
                                marker_size=10, marker_color='red', 
                                name="Optimized"), row=1, col=2)
        
        fig.update_xaxes(title_text="Dimension 1", row=1, col=2)
        fig.update_yaxes(title_text="Dimension 2", row=1, col=2)
    else:
        # Plot for fixed dimensions
        fixed_dimensions = results['optimized_dimensions']
        fig.add_trace(go.Scatter(x=[fixed_dimensions[0]], 
                                y=[fixed_dimensions[1]], 
                                mode='markers', marker_symbol='star', 
                                marker_size=10, marker_color='red', 
                                name="Fixed Dimensions"), row=1, col=2)
        
        fig.update_xaxes(title_text="Dimension 1", row=1, col=2)
        fig.update_yaxes(title_text="Dimension 2", row=1, col=2)
        fig.update_layout(xaxis2=dict(range=[1.85, 2.1]), yaxis2=dict(range=[1.85, 2.1]))

    # Dimension box plots
    piece_colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)']
    package_colors = ['rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)']
    
    for i, (dim, name) in enumerate([
        (sampled_dimensions[:, 0], "Piece Length"),
        (sampled_dimensions[:, 1], "Piece Width"),
        (sampled_dimensions[:, 2], "Piece Height")
    ]):
        fig.add_trace(go.Box(y=dim, name=name, marker_color=piece_colors[i]), row=2, col=1)
    
    for i, (dim, name) in enumerate([
        (package_length, "Package Height"),
        (package_width, "Package Length"),
        (package_height, "Package Width")
    ]):
        fig.add_trace(go.Box(y=dim, name=name, marker_color=package_colors[i]), row=2, col=1)

    # Sample data for the 3D scatter plot
    sample_size = 1000  # Limit the number of points to 1000
    sample_indices = random.sample(range(len(densities)), sample_size)
    sampled_volumes = [volumes[i] for i in sample_indices]
    sampled_densities = [densities[i] for i in sample_indices]
    sampled_masses = [individual_masses[i] for i in sample_indices]

    # Mass vs. Density vs. Volume scatter plot (3D)
    fig.add_trace(go.Scatter3d(
        x=sampled_densities, 
        y=sampled_masses, 
        z=sampled_volumes,
        mode='markers',
        marker=dict(
            size=3,
            color=sampled_masses,  # Color by mass
            colorscale='Viridis',  # Colorscale
            opacity=0.5
        ),
        name="Generated Mass vs. Density vs. Volume",
        hovertemplate=(
        "<b>Density:</b> %{x}<br>" +
        "<b>Mass:</b> %{y}<br>" +
        "<b>Volume:</b> %{z}<br>" +
        "<extra></extra>"  # Removes the trace name from appearing in the hover box
        )
    ), row=2, col=2)
    
    # Update the 3D axis labels
    fig.update_scenes(
        xaxis_title="Density (g/cm³)",
        yaxis_title="Mass (g)",
        zaxis_title="Volume (cm³)",
        row=2, col=2
    )

    # Update layout for better readability
    fig.update_layout(
        height=900,  # Increase height for better visibility
        width=1000,  # Increase width for better visibility
        boxmode='group',  # Group box plots
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,  # Move legend below the plots
            xanchor="center",
            x=0.5,
        ),
        annotations=[
            dict(
                x=0.25, y=1.0,
                xref="paper", yref="paper",
                text="Package Mass Distribution",
                showarrow=False
            ),
            dict(
                x=0.75, y=1.0,
                xref="paper", yref="paper",
                text="Dimensions",
                showarrow=False
            ),
            dict(
                x=0.25, y=0.45,
                xref="paper", yref="paper",
                text="Piece and Package Dimensions",
                showarrow=False
            ),
            dict(
                x=0.75, y=0.45,
                xref="paper", yref="paper",
                text="Mass vs. Density vs. Volume (Individual Pieces)",
                showarrow=False
            )
        ]
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Mass (g)", row=1, col=1)
    fig.update_yaxes(title_text="Dimension (cm)", row=2, col=1)

    # Update x-axis labels
    fig.update_xaxes(title_text="Mass (g)", row=1, col=1)

    return fig


# Function to create a more subtle color gradient
def create_color_gradient(hex_color, n):
    rgb = tuple(int(hex_color[1:][i:i+2], 16) for i in (0, 2, 4))
    hsv = colorsys.rgb_to_hsv(*[x/255.0 for x in rgb])
    return [colorsys.hsv_to_rgb(
        hsv[0],
        max(0, hsv[1] - 0.1 * i),  # Reduce saturation more gradually
        min(1, hsv[2] + 0.1 * i)   # Increase brightness more gradually
    ) for i in range(n)]

# Define color map to match the image
color_map = {
    'Cherry': '#b11226',
    'Fruit Punch': '#e3256b',
    'Lemon': '#fde047',
    'Orange': '#f97316',
    'Strawberry': '#f472b6',
    'Watermelon': '#ffc0cb'
}

neutral_color = '#6530c9'  # Gray color for when no flavor is selected

# Dashboard page
def dashboard_page():
    st.title("Density")

    # Sidebar for filtering
    st.sidebar.header("Filters")
    
    flavor_filter = st.sidebar.multiselect("Select Flavors", df['Flavor'].unique())
    kind_filter = st.sidebar.multiselect("Select Kinds", df['Kind'].unique())
    date_filter = st.sidebar.date_input("Select Date Range", [df['Date'].min().date(), df['Date'].max().date()])

    # Apply filters
    filtered_df = df[
        (df['Flavor'].isin(flavor_filter) if flavor_filter else True) &
        (df['Kind'].isin(kind_filter) if kind_filter else True) &
        (df['Date'].dt.date >= date_filter[0]) &
        (df['Date'].dt.date <= date_filter[1])
    ]
      # Define color map to match the image
    color_map = {
            'Cherry': '#b11226',
            'Fruit Punch': '#e3256b',
            'Lemon': '#fde047',
            'Orange': '#f97316',
            'Strawberry': '#f472b6',
            'Watermelon': '#ffc0cb'
        }

    # Determine chart color
    if len(flavor_filter) == 1:
        chart_color = color_map[flavor_filter[0]]
    else:
        chart_color = neutral_color


    # Display filtered data
    st.sidebar.subheader("Filtered Data")
    st.sidebar.dataframe(filtered_df)

   # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Samples", len(filtered_df))
        daily_count = filtered_df.groupby('Date').size().cumsum().reset_index()
        daily_count = daily_count.rename(columns={0: 'Sample Count'})  # Rename the column
        fig = px.line(daily_count, x='Date', y='Sample Count', title="Cumulative Samples Over Time")
        fig.update_traces(line_color=chart_color)
        fig.update_layout(yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        


    
    with col2:
        avg_density = filtered_df.groupby('Flavor')['Density'].mean().sort_values(ascending=False).reset_index()

        fig = px.bar(avg_density, x='Flavor', y='Density', 
                    title="Average Density by Flavor",
                    color='Flavor',
                    color_discrete_map=color_map)

        fig.update_layout(
            yaxis_title="Density (g/cm³)",
            yaxis=dict(
                range=[1.37, 1.41],  # Adjust this range as needed to show differences clearly
                dtick=0.005,  # Set tick interval to 0.005
            ),
            xaxis_title="",
            legend_title_text=""
        )

        # Add value labels on top of each bar
        fig.update_traces(texttemplate='%{y:.4f}', textposition='outside')

        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(filtered_df, x='Density', color='Flavor', title="Density Distribution by Flavor",
                    color_discrete_map=color_map)
        fig.update_layout(barmode='stack', xaxis_title="Density (g/cm³)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Create the box plot
        fig = go.Figure()

        all_kinds = sorted(filtered_df['Kind'].unique())
        flavors = sorted(filtered_df['Flavor'].unique())

        for i, kind in enumerate(all_kinds):
            for j, flavor in enumerate(flavors):
                flavor_kind_data = filtered_df[(filtered_df['Flavor'] == flavor) & (filtered_df['Kind'] == kind)]
                if not flavor_kind_data.empty:
                    base_color = color_map.get(flavor, '#000000')  # Default to black if flavor not in color_map2
                    colors = create_color_gradient(base_color, len(all_kinds))
                    fig.add_trace(go.Box(
                        x=[flavor] * len(flavor_kind_data),
                        y=flavor_kind_data['Density'],
                        name=kind,
                        legendgroup=kind,
                        offsetgroup=kind,
                        marker_color='rgba({}, {}, {}, 0.7)'.format(*[int(x * 255) for x in colors[i]]),
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8,
                        hovertext=[f"{flavor} - {kind}" for _ in range(len(flavor_kind_data))],
                        showlegend=j == 0  # Show in legend only for the first flavor
                    ))

        # Add colored rectangles behind x-axis labels
        for flavor in flavors:
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=flavors.index(flavor) - 0.4,
                y0=0,
                x1=flavors.index(flavor) + 0.4,
                y1=-0.12,
                fillcolor=color_map.get(flavor, '#000000'),
                layer="below",
                line_width=0,
            )

        # Update the layout
        fig.update_layout(
            title="Density Distribution by Flavor and Kind",
            xaxis_title="Flavor",
            yaxis_title="Density (g/cm³)",
            boxmode='group',
            legend_title="Kind",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                tickangle=0,
                tickfont=dict(color='white'),  # Set tick label color to white for contrast
                tickmode='array',
                tickvals=list(range(len(flavors))),
                ticktext=flavors
            ),
            margin=dict(b=80)  # Increase bottom margin to accommodate colored rectangles
        )

        # Adjust y-axis to show differences more clearly
        fig.update_yaxes(
            range=[filtered_df['Density'].min() - 0.01, filtered_df['Density'].max() + 0.01],
            dtick=0.005
        )

        st.plotly_chart(fig, use_container_width=True)
    

    # Calculate control limits
    mean = filtered_df['Density'].mean()
    std = filtered_df['Density'].std()
    ucl = mean + 3 * std
    lcl = mean - 3 * std

    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Individual Measurements", "Moving Range"))

    # Individual measurements
    fig.add_trace(go.Scatter(y=filtered_df['Density'], mode='markers+lines', name='Density', line=dict(color=chart_color)),
                row=1, col=1)
    fig.add_hline(y=mean, line_dash="dash", line_color="green", annotation_text="Mean", row=1, col=1)
    fig.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text="UCL", row=1, col=1)
    fig.add_hline(y=lcl, line_dash="dash", line_color="red", annotation_text="LCL", row=1, col=1)

    # Moving Range
    moving_range = np.abs(filtered_df['Density'].diff())
    mr_mean = moving_range.mean()
    mr_ucl = mr_mean + 3 * moving_range.std()

    fig.add_trace(go.Scatter(y=moving_range, mode='markers+lines', name='Moving Range', line=dict(color=chart_color)),
                row=1, col=2)
    fig.add_hline(y=mr_mean, line_dash="dash", line_color="green", annotation_text="MR Mean", row=1, col=2)
    fig.add_hline(y=mr_ucl, line_dash="dash", line_color="red", annotation_text="MR UCL", row=1, col=2)

    fig.update_layout(height=600, title_text="Density Control Chart")
    fig.update_xaxes(title_text="Sample Number", row=1, col=2)
    fig.update_xaxes(title_text="Sample Number", row=1, col=1)
    fig.update_yaxes(title_text="Density (g/cm³)", row=1, col=2)
    fig.update_yaxes(title_text="Density (g/cm³)", row=1, col=1)
    fig.update_yaxes(title_text="Moving Range", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Store the filtered data in session state for use in the optimizer
    st.session_state['filtered_data'] = filtered_df

# New X-bar and R charts
    st.subheader("X-bar and R Charts (3 Pieces, Time-based)")

    # Filter data for 3 pieces and non-null time
    xbar_df = filtered_df[(filtered_df['Kind'] == '3 Pieces') & (filtered_df['Time'].notna())]

    # Ensure Time is in the correct format
    xbar_df['Time'] = pd.to_datetime(xbar_df['Time'], format='%H:%M:%S').dt.time
    
    xbar_df = xbar_df.sort_values(['Date', 'Time'])

    # Group by Date and Time to create subgroups
    xbar_df['Subgroup'] = xbar_df.groupby(['Date', 'Time']).ngroup()
    
    # Calculate subgroup means and ranges
    subgroup_stats = xbar_df.groupby('Subgroup').agg({
        'Density': ['mean', lambda x: x.max() - x.min(), 'count'],
        'Date': 'first',
        'Time': 'first'
    }).reset_index()
    subgroup_stats.columns = ['Subgroup', 'Mean', 'Range', 'Size', 'Date', 'Time']

    # Format Date and Time
    subgroup_stats['FormattedDate'] = subgroup_stats['Date'].dt.strftime('%Y-%m-%d')
    subgroup_stats['FormattedTime'] = subgroup_stats['Time'].apply(lambda x: x.strftime('%H:%M'))
    

    # Calculate control limits
    overall_mean = np.average(subgroup_stats['Mean'], weights=subgroup_stats['Size'])
    overall_range = np.average(subgroup_stats['Range'], weights=subgroup_stats['Size'])
    
    # Function to get control chart constants
    def get_control_constants(n):
        constants = {
            2: (1.880, 0, 3.268),
            3: (1.023, 0, 2.574),
            4: (0.729, 0, 2.282),
            5: (0.577, 0, 2.114),
            6: (0.483, 0, 2.004),
            7: (0.419, 0.076, 1.924),
            8: (0.373, 0.136, 1.864),
            9: (0.337, 0.184, 1.816),
            10: (0.308, 0.223, 1.777)
        }
        return constants.get(n, constants[3])  # Default to n=3 if not in the dictionary

    # Calculate individual control limits for each subgroup
    subgroup_stats['A2'], subgroup_stats['D3'], subgroup_stats['D4'] = zip(*subgroup_stats['Size'].map(get_control_constants))
    subgroup_stats['UCL_X'] = overall_mean + subgroup_stats['A2'] * overall_range
    subgroup_stats['LCL_X'] = overall_mean - subgroup_stats['A2'] * overall_range
    subgroup_stats['UCL_R'] = subgroup_stats['D4'] * overall_range
    subgroup_stats['LCL_R'] = subgroup_stats['D3'] * overall_range

    # Create X-bar chart
    fig_xbar = go.Figure()
    fig_xbar.add_trace(go.Scatter(
        x=subgroup_stats['Subgroup'],
        y=subgroup_stats['Mean'],
        mode='lines+markers',
        name='Subgroup Mean',
        line=dict(color=chart_color),
        text=[f"{d}<br>{t}" for d, t in zip(subgroup_stats['FormattedDate'], subgroup_stats['FormattedTime'])],
        hovertemplate='%{text}<br>Mean: %{y:.4f}'
    ))
    fig_xbar.add_trace(go.Scatter(x=subgroup_stats['Subgroup'], y=subgroup_stats['UCL_X'], mode='lines', name='UCL', line=dict(dash='dash', color='red')))
    fig_xbar.add_trace(go.Scatter(x=subgroup_stats['Subgroup'], y=subgroup_stats['LCL_X'], mode='lines', name='LCL', line=dict(dash='dash', color='red')))
    fig_xbar.add_hline(y=overall_mean, line_dash="dash", line_color="green", annotation_text="Mean")

    fig_xbar.update_layout(
        title="X-bar Chart",
        xaxis=dict(
            title="Subgroup",
            tickmode='array',
            tickvals=subgroup_stats['Subgroup'],
            ticktext=[f"{d}<br>{t}" for d, t in zip(subgroup_stats['FormattedDate'], subgroup_stats['FormattedTime'])],
            tickangle=45
        ),
        yaxis_title="Subgroup Mean Density (g/cm³)",
        height=500,
        margin=dict(b=100)  # Increase bottom margin to accommodate labels
    )

    # Create R chart
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(
        x=subgroup_stats['Subgroup'],
        y=subgroup_stats['Range'],
        mode='lines+markers',
        name='Subgroup Range',
        line=dict(color=chart_color),
        text=[f"{d}<br>{t}" for d, t in zip(subgroup_stats['FormattedDate'], subgroup_stats['FormattedTime'])],
        hovertemplate='%{text}<br>Range: %{y:.4f}'
    ))
    fig_r.add_trace(go.Scatter(x=subgroup_stats['Subgroup'], y=subgroup_stats['UCL_R'], mode='lines', name='UCL', line=dict(dash='dash', color='red')))
    fig_r.add_trace(go.Scatter(x=subgroup_stats['Subgroup'], y=subgroup_stats['LCL_R'], mode='lines', name='LCL', line=dict(dash='dash', color='red')))
    fig_r.add_hline(y=overall_range, line_dash="dash", line_color="green", annotation_text="Mean Range")

    fig_r.update_layout(
        title="R Chart",
        xaxis=dict(
            title="Subgroup",
            tickmode='array',
            tickvals=subgroup_stats['Subgroup'],
            ticktext=[f"{d}<br>{t}" for d, t in zip(subgroup_stats['FormattedDate'], subgroup_stats['FormattedTime'])],
            tickangle=45
        ),
        yaxis_title="Subgroup Range",
        height=500,
        margin=dict(b=100)  # Increase bottom margin to accommodate labels
    )

    # Display X-bar and R charts
    st.plotly_chart(fig_xbar, use_container_width=True)
    st.plotly_chart(fig_r, use_container_width=True)
    st.subheader("Off the line 3-piece Sampling Control Chart Data")
    st.dataframe(xbar_df)

    # Add a button to navigate to the optimizer page
    if st.sidebar.button("Go to Optimizer"):
        st.session_state['page'] = 'optimizer'
        st.rerun()

# Optimizer page
def optimizer_page():
    st.title("Dimension Optimizer")

    # Retrieve filtered data from session state
    filtered_df = st.session_state.get('filtered_data', pd.DataFrame())

    if filtered_df.empty:
        st.warning("No data available. Please go back to the dashboard and apply filters.")
        if st.button("Back to Dashboard"):
            st.session_state['page'] = 'dashboard'
            st.rerun()
        return

    # Calculate density statistics from filtered data
    density_center = filtered_df['Density'].mean()
    density_variation = filtered_df['Density'].std()

    
    st.sidebar.subheader("Optimization Parameters")
        # Add an expander for the explanation
    # Add custom CSS for styling (same as before)
    st.sidebar.markdown("""
    <style>
    .explanation-text {
        background-color: #f0f8ff;
        border-left: 5px solid #4682b4;
        padding: 10px;
        border-radius: 5px;
    }
    .section-header {
        color: #4169e1;
        font-weight: bold;
        font-size: 1.2em;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .graph-explanation {
        background-color: #e6f3ff;
        border: 1px solid #b0d4ff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .key-point {
        color: #008080;
        font-weight: bold;
    }
    .color-example {
        display: inline-block;
        width: 15px;
        height: 15px;
        margin-right: 5px;
        border: 1px solid #000;
    }
    </style>
    """, unsafe_allow_html=True)

    # Add an expander for the explanation
    with st.sidebar.expander("📊 Simulation Explanation", expanded=False):
        st.markdown("""
        <div class="explanation-text">
        <h3 style="color: #1e90ff;">Dimension Optimizer</h3>
        <p>This tool utilizes Monte Carlo simulation to optimize dimensions for production, balancing various constraints and objectives.</p>

        <!-- Previous sections (Dimension Locking Feature, Mass Distribution Histogram, Tested Dimensions Plot) remain the same -->
        
        <div class="graph-explanation">
        <strong style="color: #ff6347;">Package Simulation</strong><br>
        <p>The simulation accounts for multiple candies per package and their arrangement:</p>
        <ul>
            <li>User defines the number of candies per package.</li>
            <li>Specs are adjusted to reflect the total package weight.</li>
            <li>Simulated masses represent the total weight of all candies in a package.</li>
            <li>Piece width (Dimension 3) contributes to the package length.</li>
            <li>Piece length (Dimension 1) contributes to the package width.</li>
            <li>Piece height (Dimension 2) contributes to the package height.</li>
        </ul>
        <p>This arrangement helps in visualizing how individual pieces impact the overall package dimensions.</p>
        </div>

        <div class="graph-explanation">
        <strong style="color: #ff6347;">Density Filtering</strong><br>
        <p>The optimization process incorporates density information from the Density page:</p>
        <ul>
            <li>The average density and standard deviation filtered on the Density page carry over to this simulation.</li>
            <li>These values are used as default inputs for the Monte Carlo simulation in this optimization process.</li>
        </ul>
        </div>

        <div class="graph-explanation">
        <strong style="color: #ff6347;">Dimension Locking Feature</strong><br>
        <p>The dimension locking feature provides users with more control over the optimization process:</p>
        <ul>
            <li>Users can lock specific dimensions to fixed values.</li>
            <li>Locking dimensions gives more flexibility in targeting specific product attributes.</li>
            <li>Without locking, the algorithm iterates through all dimensions at its own pace.</li>
            <li>Locking allows users to optimize around known constraints or preferences.</li>
            <li>This feature is useful when certain dimensions are critical for packaging or manufacturing processes.</li>
        </ul>
        </div>
                    
        <div class="graph-explanation">
            <strong style="color: #ff6347;">3D Mass-Density-Volume Plot</strong><br>
            Represents the relationship between mass, density, and volume:
            <ul>
                <li>Each point represents a simulated candy piece.</li>
                <li>Color gradient: Based on the 'viridis' colorscale.</li>
                <li><span class="color-example" style="background-color: #440154;"></span> Dark purple: Lower values</li>
                <li><span class="color-example" style="background-color: #fde725;"></span> Yellow: Higher values</li>
            </ul>
            <p>Useful for identifying trends, outliers, and relationships between these properties.</p>
        </div>

        <div class="section-header">Optimization Objectives and Key Concepts</div>
        <p>The algorithm aims to:</p>
        <ol>
            <li class="key-point">Minimize deviation from target mass</li>
            <li class="key-point">Maximize the percentage of samples above the declared weight</li>
            <li class="key-point">Optimize process capability (Cpm) for consistent production</li>
        </ol>

        <div class="graph-explanation">
            <strong style="color: #ff6347;">Mean Acceptance Value (MAV)</strong><br>
            <p>The MAV is a critical concept in our optimization process:</p>
            <ul>
                <li>It represents the lowest acceptable mass for an individual candy piece.</li>
                <li>MAV is based on standards set by NIST (National Institute of Standards and Technology).</li>
                <li>In our simulation, MAV serves as the lower specification limit.</li>
            </ul>
        </div>
        

        <div class="graph-explanation">
            <strong style="color: #ff6347;">Declared Weight Rule and Target Specification</strong><br>
            <p>The target specification is set to ensure compliance with the declared weight rule.</p>
            <p>By having the production centered at half a standard deviation above the declared, we can ensure that at least 50% of the product is above declared: </p>
            <ul>
                <li>Declared Mass: The stated weight of the product (user input).</li>
                <li>Simulated Standard Deviation (σ): Calculated from the Monte Carlo simulation.</li>
                <li>Target Spec = Declared Mass + 0.5σ</li>
            </ul>
            <p>This approach ensures:</p>
            <ul>
                <li>A certain percentage of candy pieces are above the declared weight in real production.</li>
                <li>The process safeguards against underweight packages while optimizing material usage.</li>
            </ul>
        </div>

        <div class="graph-explanation">
        <strong style="color: #ff6347;">Package Simulation</strong><br>
        <p>The simulation also accounts for multiple candies per package:</p>
        <ul>
            <li>User defines the number of candies per package.</li>
            <li>Specs are adjusted to reflect the total package weight.</li>
            <li>Simulated masses represent the total weight of all candies in a package.</li>
        </ul>
        </div>
                    
        <div class="graph-explanation">
            <strong style="color: #ff6347;">Specification Limits</strong><br>
            <p>The specification limits are:</p>
            <ul>
                <li>Lower Spec Limit (LSL) = MAV * Number of Pieces</li>
                <li>Declared Package Mass = Declared Mass per Candy * Number of Pieces</li>
                <li>Target Package Spec = Declared Package Mass + 0.5σ of Package Mass</li>
                <li>Upper Spec Limit (USL) = Target Package Spec + (Target Package Spec - LSL)</li>
            </ul>
            <p>This setup ensures that:</p>
            <ul>
                <li>No individual piece falls below the MAV.</li>
                <li>The process is centered above the declared mass to meet regulatory requirements.</li>
                <li>There's a balance between regulatory compliance and efficient material usage.</li>
            </ul>
        </div>

        <p>By considering these factors, the optimization process seeks to find dimensions that result in a production process that is compliant with regulations, efficient in material usage, and consistent in quality.</p>
        </div>
        """, unsafe_allow_html=True)

    # Add number of candies per package input
    candies_per_package = st.sidebar.number_input('Number of Pieces per Package', min_value=1, max_value=100, value=12, step=1)
    density_center = st.sidebar.number_input('Density Center', value=density_center, format="%.6f")
    density_variation = st.sidebar.number_input('Density Variation', value=density_variation, format="%.6f")
    lower_spec = st.sidebar.number_input('Lower Spec Limit', value=4.44, format="%.4f")
    declared_mass = st.sidebar.number_input('Declared Mass', value=4.89, format="%.4f")
    
    # Create columns for each dimension with their respective checkboxes
    dimcol1, dimcol2, dimcol3 = st.columns(3)

    with dimcol1:
        # Checkbox next to Dimension 1 input
        fix_dim1 = st.sidebar.checkbox("Fix", key="fix_dim1")
        dim1 = st.sidebar.number_input('Dimension 1', value=2.0, min_value=1.85, max_value=2.1, format="%.4f", disabled=not fix_dim1)

    with dimcol2:
        # Checkbox next to Dimension 2 input
        fix_dim2 = st.sidebar.checkbox("Fix", key="fix_dim2")
        dim2 = st.sidebar.number_input('Dimension 2', value=2.0, min_value=1.85, max_value=2.1, format="%.4f", disabled=not fix_dim2)

    with dimcol3:
        # Checkbox next to Dimension 3 input
        fix_dim3 = st.sidebar.checkbox("Fix", key="fix_dim3")
        dim3 = st.sidebar.number_input('Dimension 3', value=1.0, min_value=0.95, max_value=1.05, format="%.4f", disabled=not fix_dim3)

    dimension_variations = [
        st.sidebar.number_input('Dim 1 Variation', value=0.02494, format="%.5f"),
        st.sidebar.number_input('Dim 2 Variation', value=0.04427, format="%.5f"),
        st.sidebar.number_input('Dim 3 Variation', value=0.03296, format="%.5f")
    ]
    
    num_simulations = st.sidebar.number_input('Simulations', value=10000, step=1000)
    dimension_min = [1.85, 1.85, 0.95]
    dimension_max = [2.1, 2.1, 1.05]
    
    historical_masses = np.random.normal(500, 20, 1000)  # Replace with actual mass data input
    historical_densities = np.random.normal(density_center, density_variation, 1000)  # Replace with actual density data input

    # Check if all dimensions are fixed
    all_dimensions_fixed = fix_dim1 and fix_dim2 and fix_dim3

    if all_dimensions_fixed:
        st.sidebar.write("All dimensions are fixed. You can run a Monte Carlo simulation with these dimensions.")
        if st.sidebar.button("Run Monte Carlo Simulation"):
            # Perform Monte Carlo simulation with fixed dimensions
            fixed_dimensions = np.array([dim1, dim2, dim3])
            sampled_dimensions, volumes, densities, individual_masses, package_dimensions, package_height, package_length, package_width, total_package_masses = monte_carlo_simulation(
                fixed_dimensions, num_simulations, 
                density_center, density_variation, 
                dimension_variations, candies_per_package
            )

            # Calculate and display results
            mean_package_mass = np.mean(total_package_masses)
            std_dev_package_mass = np.std(total_package_masses)
            declared_package_mass = declared_mass * candies_per_package
            target_package_spec = declared_package_mass + 0.5 * std_dev_package_mass
            lower_package_spec = lower_spec * candies_per_package
            upper_package_spec = target_package_spec + (target_package_spec - lower_package_spec)

            within_spec = np.sum((total_package_masses >= lower_package_spec) & (total_package_masses <= upper_package_spec)) / len(total_package_masses)

            # Calculate Cpm
            cpm = (upper_package_spec - lower_package_spec) / (6 * np.sqrt(std_dev_package_mass**2 + (mean_package_mass - target_package_spec)**2))

            # Create and display plots
            fig = create_optimization_plots(
                {'optimized_dimensions': fixed_dimensions, 'lower_spec': lower_spec, 'declared_mass': declared_mass},
                candies_per_package,
                sampled_dimensions,
                volumes,
                densities,
                individual_masses,
                package_length,
                package_width,
                package_height,
                total_package_masses
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display results
            st.subheader("Monte Carlo Simulation Results")
            results_col, metrics_col1, metrics_col2 = st.columns(3)
            # Define colors
            length_color = "#FF6347"  # Tomato
            width_color = "#4682B4"   # Steel Blue
            height_color = "#32CD32"  # Lime Green

        
          
            with results_col:
            # Custom CSS for coloring and styling
                st.markdown("""
                <style>
                .dimension-text {
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 10px;
                }
                .dim-length { color: """ + length_color + """; }
                .dim-width { color: """ + width_color + """; }
                .dim-height { color: """ + height_color + """; }
                .avg-dimensions {
                    font-size: 16px;
                    font-weight: bold;
                    margin-top: 15px;
                    padding: 10px;
                    border-radius: 5px;
                }
                .avg-dimensions-label {
                    margin-bottom: 5px;
                }
                .avg-dimensions-values {
                    font-size: 24px;
                    margin-top: 5px;
                }
                </style>
                """, unsafe_allow_html=True)


                st.markdown(f"<div class='dimension-text dim-width'>Dim 1 (cm) - Width: {dim1:.4f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='dimension-text dim-height'>Dim 2 (cm) - Length: {dim2:.4f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='dimension-text dim-length'>Dim 3 (cm) - Height: {dim3:.4f}</div>", unsafe_allow_html=True)
                st.markdown(
                     f"<div class='avg-dimensions'>"
                    f"<div class='avg-dimensions-label'>Avg Package Dimensions "
                    f"(<span class='dim-length'>L</span> x <span class='dim-width'>W</span> x <span class='dim-height'>H</span>):</div>"
                    f"<div class='avg-dimensions-values'>"
                    f"<span class='dim-length'>{np.mean(package_width):.2f}</span> x "
                    f"<span class='dim-width'>{np.mean(package_height):.2f}</span> x "
                    f"<span class='dim-height'>{np.mean(package_length):.2f}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

            with metrics_col1:
                st.metric("Out of Spec", f"{within_spec * 100:.2f}%")
                st.metric("Mean Package Mass", f"{mean_package_mass:.4f}")
                st.metric("Std Dev Package Mass", f"{std_dev_package_mass:.4f}")

            with metrics_col2:
                st.metric("Target Package Spec", f"{target_package_spec:.4f}")
                st.metric("Process Capability (Cpm)", f"{cpm:.4f}")
                

            # Display data summaries
            data_col, data_col2 = st.columns(2)
            with data_col:
                st.subheader('Generated Data Summary')
                data_summary = pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Piece Width': [sampled_dimensions[:, 0].mean(), np.median(sampled_dimensions[:, 0]), sampled_dimensions[:, 0].std(), sampled_dimensions[:, 0].min(), sampled_dimensions[:, 0].max()],
                    'Piece Length': [sampled_dimensions[:, 1].mean(), np.median(sampled_dimensions[:, 1]), sampled_dimensions[:, 1].std(), sampled_dimensions[:, 1].min(), sampled_dimensions[:, 1].max()],
                    'Piece Height': [sampled_dimensions[:, 2].mean(), np.median(sampled_dimensions[:, 2]), sampled_dimensions[:, 2].std(), sampled_dimensions[:, 2].min(), sampled_dimensions[:, 2].max()],
                    'Piece Mass': [individual_masses.mean(), np.median(individual_masses), individual_masses.std(), individual_masses.min(), individual_masses.max()],
                    'Piece Density': [densities.mean(), np.median(densities), densities.std(), densities.min(), densities.max()],
                    'Package Length': [package_length.mean(), np.median(package_length), package_length.std(), package_length.min(), package_length.max()],
                    'Package Width': [package_width.mean(), np.median(package_width), package_width.std(), package_width.min(), package_width.max()],
                    'Package Height': [package_height.mean(), np.median(package_height), package_height.std(), package_height.min(), package_height.max()],
                    'Package Mass': [total_package_masses.mean(), np.median(total_package_masses), total_package_masses.std(), total_package_masses.min(), total_package_masses.max()]
                })
                st.dataframe(data_summary)
            with data_col2:
                st.subheader('Sample of Generated Data')
                if 'sample_rows' not in st.session_state:
                    st.session_state.sample_rows = 25
                st.session_state.sample_rows = st.number_input(
                        "Number of rows", 
                        min_value=5, 
                        max_value=len(sampled_dimensions), 
                        value=st.session_state.sample_rows,
                        step=5)
                sample_data = pd.DataFrame({
                    'Piece Width': sampled_dimensions[:st.session_state.sample_rows, 0],
                    'Piece Length': sampled_dimensions[:st.session_state.sample_rows, 1],
                    'Piece Height': sampled_dimensions[:st.session_state.sample_rows, 2],
                    'Piece Volume': volumes[:st.session_state.sample_rows],
                    'Piece Density': densities[:st.session_state.sample_rows],
                    'Piece Mass': individual_masses[:st.session_state.sample_rows],
                    'Package Length': package_length[:st.session_state.sample_rows],
                    'Package Width': package_width[:st.session_state.sample_rows],
                    'Package Height': package_height[:st.session_state.sample_rows],
                    'Package Mass': total_package_masses[:st.session_state.sample_rows]
                })
                st.write(f"Displaying {len(sample_data)} rows out of {num_simulations} total rows")
                st.dataframe(sample_data)

    else:

        if st.sidebar.button("Run Optimization"):
            # Determine which dimensions to optimize
            dims_to_optimize = [not fix_dim1, not fix_dim2, not fix_dim3]
            fixed_dims = [dim1 if fix_dim1 else None, 
                        dim2 if fix_dim2 else None, 
                        dim3 if fix_dim3 else None]
            
            # Perform optimization
            result, tested_dimensions = perform_optimization(dims_to_optimize,
                                                            fixed_dims,
                                                            dimension_min,
                                                            dimension_max,
                                                            num_simulations, 
                                                            density_center, 
                                                            density_variation, 
                                                            dimension_variations, 
                                                            lower_spec, 
                                                            declared_mass,
                                                            candies_per_package)
            
            # Combine optimized and fixed dimensions
            optimized_dimensions = []
            opt_index = 0
            for i, fixed in enumerate(fixed_dims):
                if fixed is not None:
                    optimized_dimensions.append(fixed)
                else:
                    optimized_dimensions.append(result.x[opt_index])
                    opt_index += 1
            
            # Store results in session state
            st.session_state['optimization_results'] = {
                'result': result,
                'tested_dimensions': tested_dimensions,
                'optimized_dimensions': np.array(optimized_dimensions),
                'num_simulations': num_simulations,
                'density_center': density_center,
                'density_variation': density_variation,
                'dimension_variations': dimension_variations,
                'lower_spec': lower_spec,
                'declared_mass': declared_mass
            }

        
            if 'optimization_results' in st.session_state:
                results = st.session_state['optimization_results']
                
                # Display optimized dimensions
                st.subheader("Optimization Results")

                # Run Monte Carlo simulation with optimized dimensions
                sampled_dimensions, volumes, densities, individual_masses, package_dimensions, package_width, package_height, package_length, total_package_masses = monte_carlo_simulation(
                    results['optimized_dimensions'], results['num_simulations'], 
                    results['density_center'], results['density_variation'], 
                    results['dimension_variations'], candies_per_package
                )

                # Calculate package masses (this step is redundant as total_package_masses is already calculated)
                # package_masses = total_package_masses  # Use this if you need package_masses elsewhere

                mean_package_mass = np.mean(total_package_masses)
                std_dev_package_mass = np.std(total_package_masses)
                declared_package_mass = results['declared_mass'] * candies_per_package
                target_package_spec = declared_package_mass + 0.5 * std_dev_package_mass
                lower_package_spec = results['lower_spec'] * candies_per_package
                upper_package_spec = target_package_spec + (target_package_spec - lower_package_spec)

                within_spec = np.sum((total_package_masses >= lower_package_spec) & (total_package_masses <= upper_package_spec)) / len(total_package_masses)

                # Calculate Cpm
                cpm = (upper_package_spec - lower_package_spec) / (6 * np.sqrt(std_dev_package_mass**2 + (mean_package_mass - target_package_spec)**2))
                
                # Use the new plotting function
                fig = create_optimization_plots(
                    st.session_state['optimization_results'], 
                    candies_per_package,
                    sampled_dimensions,
                    volumes,
                    densities,
                    individual_masses,
                    package_length,
                    package_width,
                    package_height,
                    total_package_masses
                )
                
                # Plot the figure
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Simulated Results")
                # Display key metrics
                results_col, metrics_col1, metrics_col2 = st.columns(3)
                # Define colors
                length_color = "#FF6347"  # Tomato
                width_color = "#4682B4"   # Steel Blue
                height_color = "#32CD32"  # Lime Green
                with results_col:
                    # Custom CSS for coloring and styling
                    st.markdown("""
                    <style>
                    .dimension-text {
                        font-size: 24px;
                        font-weight: bold;
                        margin-bottom: 10px;
                    }
                    .dim-length { color: """ + length_color + """; }
                    .dim-width { color: """ + width_color + """; }
                    .dim-height { color: """ + height_color + """; }
                    .avg-dimensions {
                        font-size: 16px;
                        font-weight: bold;
                        margin-top: 15px;
                        padding: 10px;
                        border-radius: 5px;
                    }
                    .avg-dimensions-label {
                        margin-bottom: 5px;
                    }
                    .avg-dimensions-values {
                        font-size: 24px;
                        margin-top: 5px;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    # uses the average of the data instead of the specified center like in all fixed dimension
                    st.markdown(f"<div class='dimension-text dim-width'>Dim 1 (cm) - Width: {sampled_dimensions[:, 0].mean():.4f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='dimension-text dim-height'>Dim 2 (cm) - Length: {sampled_dimensions[:, 1].mean():.4f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='dimension-text dim-length'>Dim 3 (cm) - Height: {sampled_dimensions[:, 2].mean():.4f}</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='avg-dimensions'>"
                        f"<div class='avg-dimensions-label'>Avg Package Dimensions "
                        f"(<span class='dim-length'>L</span> x <span class='dim-width'>W</span> x <span class='dim-height'>H</span>):</div>"
                        f"<div class='avg-dimensions-values'>"
                        f"<span class='dim-length'>{np.mean(package_length):.2f}</span> x "
                        f"<span class='dim-width'>{np.mean(package_width):.2f}</span> x "
                        f"<span class='dim-height'>{np.mean(package_height):.2f}</span>"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )
                
                # Update the metrics display
                with metrics_col1:
                    st.metric("Above MAV", f"{within_spec * 100:.2f}%")
                    st.metric("Mean Package Mass", f"{mean_package_mass:.4f}")
                    st.metric("Std Dev Package Mass", f"{std_dev_package_mass:.4f}")
                with metrics_col2:
                    st.metric("Target Package Spec", f"{target_package_spec:.4f}")
                    st.metric("Process Capability (Cpm)", f"{cpm:.4f}")
            
                data_col, data_col2 = st.columns(2)
                with data_col:
                    st.subheader('Generated Data Summary')
                    data_summary = pd.DataFrame({
                        'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                        'Piece Width': [sampled_dimensions[:, 0].mean(), np.median(sampled_dimensions[:, 0]), sampled_dimensions[:, 0].std(), sampled_dimensions[:, 0].min(), sampled_dimensions[:, 0].max()],
                        'Piece Length': [sampled_dimensions[:, 1].mean(), np.median(sampled_dimensions[:, 1]), sampled_dimensions[:, 1].std(), sampled_dimensions[:, 1].min(), sampled_dimensions[:, 1].max()],
                        'Piece Height': [sampled_dimensions[:, 2].mean(), np.median(sampled_dimensions[:, 2]), sampled_dimensions[:, 2].std(), sampled_dimensions[:, 2].min(), sampled_dimensions[:, 2].max()],
                        'Piece Mass': [individual_masses.mean(), np.median(individual_masses), individual_masses.std(), individual_masses.min(), individual_masses.max()],
                        'Piece Density': [densities.mean(), np.median(densities), densities.std(), densities.min(), densities.max()],
                        'Package Length': [package_length.mean(), np.median(package_length), package_length.std(), package_length.min(), package_length.max()],
                        'Package Width': [package_width.mean(), np.median(package_width), package_width.std(), package_width.min(), package_width.max()],
                        'Package Height': [package_height.mean(), np.median(package_height), package_height.std(), package_height.min(), package_height.max()],
                        'Package Mass': [total_package_masses.mean(), np.median(total_package_masses), total_package_masses.std(), total_package_masses.min(), total_package_masses.max()]
                    })

                    st.dataframe(data_summary)
                with data_col2:
                    st.subheader('Sample of Generated Data')
                    
                    # Initialize the number of rows in session state if not already present
                    if 'sample_rows' not in st.session_state:
                        st.session_state.sample_rows = 25

                    st.session_state.sample_rows = st.number_input(
                            "Number of rows", 
                            min_value=5, 
                            max_value=len(sampled_dimensions), 
                            value=st.session_state.sample_rows,
                            step=5)
                    
                    # Display the sample data with the adjusted number of rows
                    sample_data = pd.DataFrame({
                        'Piece Length': sampled_dimensions[:st.session_state.sample_rows, 0],
                        'Piece Width': sampled_dimensions[:st.session_state.sample_rows, 1],
                        'Piece Height': sampled_dimensions[:st.session_state.sample_rows, 2],
                        'Piece Volume': volumes[:st.session_state.sample_rows],
                        'Piece Density': densities[:st.session_state.sample_rows],
                        'Piece Mass': individual_masses[:st.session_state.sample_rows],
                        'Package Length': package_length[:st.session_state.sample_rows],
                        'Package Width': package_width[:st.session_state.sample_rows],
                        'Package Height': package_height[:st.session_state.sample_rows],
                        'Package Mass': total_package_masses[:st.session_state.sample_rows]
                    })
                    st.write(f"Displaying {len(sample_data)} rows out of {num_simulations} total rows")
                    st.dataframe(sample_data)
                

        
# Main app logic
def main():
    st.set_page_config(layout="wide")

    # Initialize session state for page navigation if not already set
    if 'page' not in st.session_state:
        st.session_state['page'] = 'dashboard'

    # Display the appropriate page based on the session state
    if st.session_state['page'] == 'dashboard':
        dashboard_page()
    elif st.session_state['page'] == 'optimizer':
        optimizer_page()

if __name__ == "__main__":
    main()