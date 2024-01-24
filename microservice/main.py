from flask import Flask, render_template, request
from food_recognizer import inference


app = Flask(__name__)


@app.route('/')
def index() -> str:
    """
    Render a template by name with the given context. Show html page

    Returns
    -------
    str
    """
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload() -> str:
    """
    Receive the file from the client. Api endpoint for image upload

    Returns
    -------
    str
    String with an url_root and filepath
    """
    file = request.files['file']
    filepath = f'static/temp/{file.filename}'
    file.save(filepath)
    inference(filepath)

    return f"{request.url_root}{filepath}"


if __name__ == '__main__':
    app.run()
