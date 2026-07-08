from flask import Flask

app = Flask(__name__)

@app.route('/')
def test_page():
    return "<h1>Great News! Python and Flask are communicating perfectly!</h1>"

if __name__ == "__main__":
    print("--- Attempting to launch server on Port 9000 ---")
    app.run(debug=True, port=9000)