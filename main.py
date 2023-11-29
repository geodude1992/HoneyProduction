"""
    Honey Production
As you may have already heard, the honeybees are in a precarious state right now.
You may have seen articles about the decline of the honeybee population for various reasons.
You want to investigate this decline and how the trends of the past predict the future for the honeybees.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import plotly.graph_objects as go
import plotly.express as px

# Dataset Honey Production in the USA (1998-2012)
# Source https://www.kaggle.com/datasets/jessicali9530/honey-production
df = pd.read_csv("honeyproduction.csv")

# print csv header to console: state, numcol ,yieldpercol, totalprod, stocks, priceperlb, prodvalue, year
print(df.head)






"""
    Calculating the average (mean) total honey production per year across all states
"""
def average_production():
    df = pd.read_csv("honeyproduction.csv")
    # average production of honey per year using .groupby() method provided by pandas to get the mean per year
    prod_per_year = df.groupby('year').totalprod.mean().reset_index()

    # total production of honey per year using .groupby() method provided by pandas to get the mean per year
    prod_per_year = df.groupby('year').totalprod.sum().reset_index()

    # columnOfInterest = df['columnName'] then reshape it to get it into the right format
    X = prod_per_year['year']
    #X1 = total_prod_per_year['year']
    X = X.values.reshape(-1, 1)
    #X1 = X1.values.reshape(-1, 1)

    # total product per year
    y = prod_per_year['totalprod']
    #y1 = total_prod_per_year['totalprod']

    # Using scatter(), plot y vs X as a scatterplot.
    """
        ScatterPlot Using Plotly    
    """
    # Create a scatter plot
    scatter = go.Scatter(x=X.ravel(), y=y, mode='markers', name='Total Production')

    # Create a figure
    fig = go.Figure(data=scatter)

    # Update layout of the figure
    fig.update_layout(
        title='Average Honey Production per Year across all states',
        #title='Total Honey Production per Year across all states',
        xaxis_title='Year',
        yaxis_title='Average Production (in pounds)',
        #yaxis_title='Average Production (in pounds)',
        hovermode='closest'
    )

    # Show the figure
    fig.show()


"""
    Calculating the Total sum honey production per year across all states
"""


def average_production():
    df = pd.read_csv("honeyproduction.csv")
    # average production of honey per year using .groupby() method provided by pandas to get the mean per year
    prod_per_year = df.groupby('year').totalprod.mean().reset_index()

    # total production of honey per year using .groupby() method provided by pandas to get the mean per year
    prod_per_year = df.groupby('year').totalprod.sum().reset_index()

    # columnOfInterest = df['columnName'] then reshape it to get it into the right format
    X = prod_per_year['year']
    # X1 = total_prod_per_year['year']
    X = X.values.reshape(-1, 1)
    # X1 = X1.values.reshape(-1, 1)

    # total product per year
    y = prod_per_year['totalprod']
    # y1 = total_prod_per_year['totalprod']

    # Using scatter(), plot y vs X as a scatterplot.
    """
        ScatterPlot Using Plotly    
    """
    # Create a scatter plot
    scatter = go.Scatter(x=X.ravel(), y=y, mode='markers', name='Total Production')

    # Create a figure
    fig = go.Figure(data=scatter)

    # Update layout of the figure
    fig.update_layout(
        title='Average Honey Production per Year across all states',
        # title='Total Honey Production per Year across all states',
        xaxis_title='Year',
        yaxis_title='Average Production (in pounds)',
        # yaxis_title='Average Production (in pounds)',
        hovermode='closest'
    )

    # Show the figure
    fig.show()




"""
    LinearRegression Using Plotly    
    Predict values based on your X (years). 
"""
# Create a linear regression model from scikit-learn
# var = module.constructor
regr = linear_model.LinearRegression()

# Fit the model to the data by using .fit() feed X into your regr model
# by passing it in as a parameter of .fit()
"""
Params:
    X – Training data.
    y – Target values. Will be cast to X's dtype if necessary.
    sample_weight – Individual weights for each sample.
Returns:
    Fitted Estimator.
"""
regr.fit(X, y, sample_weight=None)

# The slope of the line will be the first (and only) element of the regr.coef_ list.
print("Slope:", regr.coef_[0])
print("Intercept:", regr.intercept_)

# Create a list called y_predict that is the predictions of the linear regression model
"""
def predict(self, X: Any) -> Any
    Predict using the linear model.
    Params:
    X – Samples.
    Returns:
    Returns predicted values.
"""
y_predict = regr.predict(X)

# Create the scatter plot for actual data
scatter = go.Scatter(x=X.ravel(), y=y, mode='markers', name='Total Production')

# Create a line plot for the predictions
line = go.Scatter(x=X.ravel(), y=y_predict, mode='lines', name='Regression Line')

# Create a figure and add both the scatter and line plots
fig = go.Figure(data=[scatter, line])

# Update layout of the figure
fig.update_layout(
    title='Average Honey Production per Year across all states Linear Model',
    xaxis_title='Year',
    yaxis_title='Average Production (in pounds)',
    hovermode='closest'
)

# Show the figure
fig.show()




"""
    Future Prediction Using Plotly   
"""
# production of honey has been in decline, according to this linear model.
# Let’s predict what the year 2050 may look like in terms of honey production.
X_future = np.array(range(2013, 2051))
# You can think of reshape() as rotating this array. Rather than one big row of numbers,
# X_future is now a big column of numbers — there’s one number in each row.
X_future = X_future.reshape(-1, 1)

# list called future_predict that is the y-values that your regr model
# would predict for the values of X_future.
future_predict = regr.predict(X_future)

# Create a line plot for the future predictions
future_line = go.Scatter(x=X_future.ravel(), y=future_predict, mode='lines', name='Future Predictions', line=dict(dash='dash'))

# Add the future predictions line to the existing figure
fig.add_trace(future_line)

# Update the figure layout to accommodate the new data
fig.update_layout(
    title='Average Honey Production Forecast (1998-2050)',
    xaxis_title='Year',
    yaxis_title='Average Production (in pounds)',
    hovermode='closest',
    paper_bgcolor="#EBA937"
)

# Show the updated figure
fig.show()




"""
    STOCK PER YEAR
"""
# Calculate mean stocks per year
stocks_per_year = df.groupby('year')['stocks'].sum().reset_index()

# Prepare the data for plotting
years = stocks_per_year['year']
stocks = stocks_per_year['stocks']

# Function to generate a gradient
def generate_gradient(start_hex, end_hex, n):
    start_rgb = np.array([int(start_hex[i:i+2], 16) for i in (0, 2, 4)])
    end_rgb = np.array([int(end_hex[i:i+2], 16) for i in (0, 2, 4)])
    return ['#' + ''.join([f'{int(i):02x}' for i in (start_rgb + (end_rgb - start_rgb) * k / (n - 1))]) for k in range(n)]

# Generate gradient colors
colors = generate_gradient('EBA937', '9B5E0F', len(years))

# Create a bar plot for stocks per year with gradient colors
stocks_bar = go.Bar(x=years, y=stocks, name='Stocks', marker=dict(color=colors))

# Create a figure and add the stocks bar plot
fig = go.Figure(data=stocks_bar)

# Update layout of the figure
fig.update_layout(
    title='Total Honey Stocks per Year (1998-2012)',
    xaxis_title='Year',
    yaxis_title='Total Stocks (in pounds)',
)

# Show the figure
fig.show()




"""
    Amount of Honey per Year by State
Production values were grouped by state and changes in production were measured over time
"""
# Calculate total honey production per year for each state
total_prod_per_year_state = df.groupby(['year', 'state'])['totalprod'].sum().reset_index()

# Prepare a list of states and a color palette
states = total_prod_per_year_state['state'].unique()
colors = px.colors.qualitative.Plotly

# Create a figure
fig = go.Figure()

# Add a line for each state
for i, state in enumerate(states):
    # Filter data for the state
    state_data = total_prod_per_year_state[total_prod_per_year_state['state'] == state]

    # Create a line plot for this state
    # The modulo operation (i % len(colors)) is used to cycle through the colors
    # if there are more states than colors in the palette.
    fig.add_trace(go.Scatter(x=state_data['year'], y=state_data['totalprod'],
                             mode='lines', name=state,
                             line=dict(color=colors[i % len(colors)])))

# Update layout of the figure
fig.update_layout(
    title='Total Honey Produced by State between 1998 and 2012',
    xaxis_title='Year',
    yaxis_title='Total Production (in pounds)',
    hovermode='closest'
)

# Show the figure
fig.show()





"""
    Average Yield per Colony Per Year Using Box Plots
    
    Box plots were invented by John Tukey, a famous statistician, in 1970. 
    He called them box-and-whisker plots, and they were part of his 
    exploratory data analysis approach.
"""
# Get unique years and sort them
years = df['year'].unique()
years.sort()

# Generate a list of colors from a (rainbow) color scale
#colors = px.colors.sequential.Rainbow
colors = px.colors.qualitative.Plotly

# Normalize the years to the range of the color scale
normalized_years = (years - years.min()) / (years.max() - years.min())
year_colors = [colors[int(i * (len(colors) - 1))] for i in normalized_years]

# Create a figure
fig = go.Figure()

# Add a box for each year with its corresponding color
for i, year in enumerate(years):
    year_data = df[df['year'] == year]
    fig.add_trace(go.Box(y=year_data['yieldpercol'], name=str(year),
                         marker_color=year_colors[i]))

# Update layout of the figure
fig.update_layout(
    title='Average Yield per Colony Per Year',
    xaxis_title='Year',
    yaxis_title='Yield per Colony',
    showlegend=False
)

# Show the figure
fig.show()




