from flask import Flask, request, render_template
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = VGG16(weights='imagenet', include_top=True)

# 画像の前処理
def process_image(img):
    img = image.load_img(img, target_size=(224, 224))  # VGG16の入力サイズにリサイズ
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# 予測関数
def predict(img):
    img = process_image(img)
    preds = model.predict(img)
    decoded_preds = decode_predictions(preds, top=5)[0]  # 上位5つのカテゴリを取得
    result = [(label, round(float(score) * 100, 2)) for (_, label, score) in decoded_preds]
    return result

# メインのルート
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        # ユーザーがファイルをアップロードした場合
        img_file = request.files['image']
        if img_file:
            img_path = f"uploads/{img_file.filename}"
            img_file.save(img_path)
            result = predict(img_path)
            return render_template('result.html', result=result, img_path=img_path)
    
    # ファイルがアップロードされていない場合やGETリクエストの場合は、アップロードページを表示
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)

