from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
model = load_model('model_without_preprocess_finetuned.h5')
plot_model(model, 'model.jpg')