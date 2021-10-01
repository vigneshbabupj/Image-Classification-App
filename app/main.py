from flask import Flask, request, jsonify, render_template

from app.torch_utils import transform_image, get_prediction

import base64

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')  
def upload():  
    return render_template("file_upload_form.html") 

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            
            data = base64.b64encode(img_bytes)
            data = data.decode() 
            img_tag = "<img src='data:image/png;base64," + data + "'/>"

            return render_template("success.html", pred = prediction, image = img_tag)
        
        except:
            return jsonify({'error': 'error during prediction'})

if __name__ == "__main__":
    app.run(debug=True)