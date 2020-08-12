# Import dash
import dash
import dash_html_components as html

# Creating app
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])

# Associating server
server = app.server

# Define title and layout
app.title = 'COVID 19 - World cases'
app.layout = html.H1("COVID 19 🦠 - Day to day evolution all over the world v2", className="header__text")

if __name__ == "__main__":
    # Starting flask server
    app.run_server(debug=True)