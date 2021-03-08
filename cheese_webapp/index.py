import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import web

render = web.template.render("cheese_webapp/")
ruteModel = 'static/model/keras_model.h5'

class Index:
    def GET(self):
        img = None
        return render.index()

    def POST(self):
        form = web.input()
        image = form['a']
        #image = Image.open("rute/")

        detection_graph = tensorflow.Graph()
        with detection_graph.as_default():
            od_graph_def = tensorflow.GraphDef()
            with tensorflow.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tensorflow.import_graph_def(od_graph_def, name='')
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        #model = tensorflow.keras.models.load_model(ruteModel)

        np.set_printoptions(suppress=True)
        model = tensorflow.keras.models.load_model(ruteModel)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        image.show()
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        print(prediction)
        return render.index(prediction)