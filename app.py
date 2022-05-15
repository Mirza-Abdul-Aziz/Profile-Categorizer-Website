from crypt import methods
import imp
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/search', methods=["POST", "GET"]) 
def search():
    return render_template("search.html")

if __name__ == "__main__":
    app.run(debug = True)