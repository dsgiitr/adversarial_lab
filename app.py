from flask import Flask, request, render_template
import base64 as b64
app = Flask(__name__ , template_folder='template')

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/hello', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        with open('images/sakshi.png', 'wb') as f:
            data = b64.b64decode(request.form['data'])
            f.write(data)
        return 'Image Saved!'

