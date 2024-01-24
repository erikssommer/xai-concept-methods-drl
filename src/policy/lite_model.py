import tensorflow as tf
import numpy as np

"""
Based on https://micwurm.medium.com/using-tensorflow-lite-to-speed-up-predictions-a3954886eb98.
"""

class LiteModel:

    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model, ))

    @classmethod
    def from_keras_model_as_bytes(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return tflite_model

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        val_output_det = self.interpreter.get_output_details()[0]
        dist_output_det = self.interpreter.get_output_details()[1]

        self.input_index = input_det["index"]
        self.dist_output_index = dist_output_det["index"]
        self.val_output_index = val_output_det["index"]
        self.input_shape = input_det["shape"]
        self.dist_output_shape = dist_output_det["shape"]
        self.val_output_shape = val_output_det["shape"]

        self.input_dtype = input_det["dtype"]
        self.dist_output_dtype = dist_output_det["dtype"]
        self.val_output_dtype = val_output_det["dtype"]

    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        dist_out = self.interpreter.get_tensor(self.dist_output_index)
        val_out = self.interpreter.get_tensor(self.val_output_index)
        return dist_out[0], val_out[0]