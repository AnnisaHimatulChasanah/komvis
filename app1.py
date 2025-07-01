import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from PIL import Image

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Deteksi Real-time dengan InceptionV3 (TFLite)",
    page_icon="📸",
    layout="centered"
)

st.title("Deteksi Gambar Real-time dengan InceptionV3 (TFLite)")
st.write("Aplikasi ini mendeteksi kelas dari gambar kamera secara real-time menggunakan model TFLite.")

# --- Load Model TFLite ---
@st.cache_resource
def load_tflite_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="model_inceptionv3_best.tflite")
        interpreter.allocate_tensors()
        st.success("✅ Model TFLite berhasil dimuat!")
        return interpreter
    except Exception as e:
        st.error(f"❌ Gagal memuat model TFLite: {e}")
        return None

interpreter = load_tflite_model()

# --- Kelas dan Ukuran Gambar ---
CLASS_NAMES = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot',
    'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus', 'healthy', 'powdery_mildew'
]
INPUT_SHAPE = (299, 299)

# --- Preprocessing Gambar ---
def preprocess_image(image_array, target_size):
    image_pil = Image.fromarray(image_array)
    image_resized = image_pil.resize(target_size)
    image_array_resized = img_to_array(image_resized)
    image_array_normalized = image_array_resized / 255.0
    image_array_expanded = np.expand_dims(image_array_normalized, axis=0)
    return image_array_expanded

# --- Prediksi dengan TFLite ---
def predict_frame_tflite(frame, interpreter, class_names, input_shape):
    preprocessed = preprocess_image(frame, input_shape)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], preprocessed.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = np.argmax(output_data[0])
    confidence = np.max(output_data[0]) * 100
    predicted_class_name = class_names[predicted_index]

    return predicted_class_name, confidence

# --- Stream dari Kamera ---
if interpreter:
    st.subheader("Ambil Gambar dari Kamera")
    st.write("Klik 'Mulai Deteksi' untuk mengaktifkan kamera dan melihat hasil deteksi real-time.")

    frame_placeholder = st.empty()
    detection_text_placeholder = st.empty()

    start_button = st.button("Mulai Deteksi")
    stop_button = st.button("Hentikan Deteksi")

    if start_button:
        st.session_state.run_camera = True
    if stop_button:
        st.session_state.run_camera = False

    if 'run_camera' not in st.session_state:
        st.session_state.run_camera = False

    if st.session_state.run_camera:

        class VideoProcessor(VideoTransformerBase):
            def __init__(self):
                self.interpreter = interpreter
                self.class_names = CLASS_NAMES
                self.input_shape = INPUT_SHAPE
                self.result_text = "Menunggu deteksi..."

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                predicted_class_name, confidence = predict_frame_tflite(
                    img_rgb, self.interpreter, self.class_names, self.input_shape
                )
                self.result_text = f"Deteksi: {predicted_class_name} ({confidence:.2f}%)"
                cv2.putText(img, self.result_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                return img

        st.info("Memulai stream kamera. Harap izinkan akses kamera di browser Anda.")

        ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            video_transformer_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
        )

        if ctx.state.playing and ctx.video_transformer:
            detection_text_placeholder.write(ctx.video_transformer.result_text)
    else:
        st.warning("Tekan 'Mulai Deteksi' untuk memulai.")

st.markdown("---")
st.write("Aplikasi ini dibuat untuk mendeteksi penyakit tanaman tomat secara real-time.")
