print('enter __init__')
import dash
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

server = Flask(__name__)
server.config['DEBUG'] = True
server.config['SQLALCHEMY_ECHO'] = True
server.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///replays.db'
server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(server)
db.session.configure(autoflush=False)
app = dash.Dash(__name__, server = server, url_base_pathname = '/dashboard/')
app.config['suppress_callback_exceptions'] = True

print('exit __init__')
