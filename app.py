from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image
from flask_cors import CORS
import mysql.connector

app = Flask(__name__)
CORS(app)

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root", 
    database="rice_classifier"
)
cursor = db.cursor(dictionary=True)

# Load model
model = YOLO('beras_clf/yolov8-cls/weights/best.pt')

CLASS_MAP = {
    0: 'Beras hitam',
    1: 'Beras ir42',
    2: 'Beras ketan',
    3: 'Beras merah',
    4: 'beras basmati',
    5: 'beras buloq',
    6: 'beras japonica'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'File name missing'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        results = model(image)
        probs = results[0].probs
        pred_idx = int(probs.top1)
        confidence = float(probs.top1conf)
        varietas = CLASS_MAP[pred_idx]

        # Ambil informasi dari DB
        cursor.execute("SELECT * FROM beras_info WHERE nama = %s", (varietas,))
        info = cursor.fetchone()

        # Simpan riwayat
        cursor.execute(
            "INSERT INTO riwayat (filename, varietas, confidence) VALUES (%s, %s, %s)",
            (file.filename, varietas, confidence)
        )
        db.commit()

        return jsonify({
            'varietas': varietas,
            'ukuran': info['ukuran'],
            'warna': info['warna'],
            'tekstur': info['tekstur'],
            'deskripsi': info['deskripsi']
        })
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
