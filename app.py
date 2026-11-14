from collections import deque
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps, ImageFilter
from ai_edge_litert.interpreter import Interpreter
import numpy as np
import os
import threading

app = Flask(__name__)

print("Cargando modelo de IA...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo.tflite")

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
lock_modelo = threading.Lock()

print("Modelo cargado exitosamente.")


def conservar_digito_principal(arr):
    img_tmp = Image.fromarray(arr.astype(np.uint8))
    w, h = img_tmp.size
    escala = 140.0 / max(w, h)
    if escala < 1.0:
        img_tmp = img_tmp.resize((max(1, round(w * escala)), max(1, round(h * escala))), Image.Resampling.BILINEAR)

    mascara_busqueda = np.array(img_tmp.filter(ImageFilter.MaxFilter(3))) > 0
    filas, cols = mascara_busqueda.shape
    visitado = np.zeros_like(mascara_busqueda, dtype=bool)

    mejor_componente = []

    for r in range(filas):
        for c in range(cols):
            if mascara_busqueda[r, c] and not visitado[r, c]:
                cola = deque([(r, c)])
                visitado[r, c] = True
                componente = []

                while cola:
                    cr, cc = cola.popleft()
                    componente.append((cr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < filas and 0 <= nc < cols and mascara_busqueda[nr, nc] and not visitado[nr, nc]:
                                visitado[nr, nc] = True
                                cola.append((nr, nc))

                if len(componente) > len(mejor_componente):
                    mejor_componente = componente

    if not mejor_componente:
        return arr

    mascara_principal = np.zeros((filas, cols), dtype=np.uint8)
    for r, c in mejor_componente:
        mascara_principal[r, c] = 255

    mascara_original = Image.fromarray(mascara_principal).resize((w, h), Image.Resampling.NEAREST)
    return np.where(np.array(mascara_original) > 0, arr, 0)


def preprocesar_imagen(stream, expected_shape):
    img_raw = Image.open(stream)
    if img_raw.mode in ('RGBA', 'LA') or (img_raw.mode == 'P' and 'transparency' in img_raw.info):
        fondo = Image.new('RGBA', img_raw.size, (255, 255, 255, 255))
        img_raw = Image.alpha_composite(fondo, img_raw.convert('RGBA'))

    img = img_raw.convert('L')

    if np.median(np.array(img)) > 127:
        img = ImageOps.invert(img)

    img = ImageOps.autocontrast(img, cutoff=2)
    arr = np.array(img, dtype=np.float32)
    arr[arr < 50] = 0
    arr = conservar_digito_principal(arr)
    img_limpia = Image.fromarray(arr.astype(np.uint8))

    bbox = img_limpia.getbbox()
    if bbox:
        digito = img_limpia.crop(bbox)
        w, h = digito.size

        escala_base = 56.0 / max(w, h)
        base_w = max(1, round(w * escala_base))
        base_h = max(1, round(h * escala_base))
        digito_base = digito.resize((base_w, base_h), Image.Resampling.LANCZOS)

        densidad = np.mean(np.array(digito_base) > 127)
        if densidad > 0.42:
            digito_base = digito_base.filter(ImageFilter.MinFilter(3))
        elif densidad < 0.22:
            digito_base = digito_base.filter(ImageFilter.MaxFilter(3))

        escala = 20.0 / max(base_w, base_h)
        nuevo_w = max(1, round(base_w * escala))
        nuevo_h = max(1, round(base_h * escala))
        digito_final = digito_base.resize((nuevo_w, nuevo_h), Image.Resampling.LANCZOS)

        lienzo = Image.new('L', (28, 28), 0)
        lienzo.paste(digito_final, ((28 - nuevo_w) // 2, (28 - nuevo_h) // 2))
    else:
        lienzo = img_limpia.resize((28, 28), Image.Resampling.LANCZOS)

    img_array = np.array(lienzo, dtype=np.float32) / 255.0

    if img_array.size != int(np.prod(expected_shape)):
        raise ValueError(
            f"Dimensiones incompatibles. Esperadas: {expected_shape}, "
            f"recibidas: {img_array.shape}"
        )

    return img_array.reshape(expected_shape)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predecir', methods=['POST'])
def predecir():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400

    archivo = request.files['imagen']

    if archivo.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    try:
        expected_shape = tuple(input_details[0]['shape'])
        img_lista = preprocesar_imagen(archivo.stream, expected_shape)

        with lock_modelo:
            interpreter.set_tensor(
                input_details[0]['index'],
                img_lista.astype(input_details[0]['dtype'])
            )
            interpreter.invoke()
            predicciones = interpreter.get_tensor(
                output_details[0]['index']
            ).copy()

        numero_predicho = int(np.argmax(predicciones[0]))
        confianza = float(np.max(predicciones[0]) * 100)

        return jsonify({
            'numero_predicho': numero_predicho,
            'confianza': round(confianza, 2)
        })

    except Exception as e:
        print(f"Error en predicción: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )