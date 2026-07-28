import tensorflow as tf


class SimCLRAugmentation:

    def __init__(self, image_size=32):

        self.image_size = image_size


    ########################################################
    # Random Resized Crop (SimCLR)
    ########################################################

    def random_resized_crop(self, image):

        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.zeros([0, 0, 4]),
            min_object_covered=0.0,
            aspect_ratio_range=(0.75, 1.33),
            area_range=(0.2, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True
        )
    
        image = tf.slice(image, bbox_begin, bbox_size)
    
        image = tf.image.resize(image, (self.image_size, self.image_size))
    
        return image


    ########################################################
    # Horizontal Flip
    ########################################################

    def horizontal_flip(self, image):

        return tf.image.random_flip_left_right(image)


    ########################################################
    # Color Jitter
    ########################################################

    def color_jitter(self, image):

        p = tf.random.uniform(())

        def apply():

            x = tf.image.random_brightness(image, 0.8)
            x = tf.image.random_contrast(x, 0.2, 1.8)
            x = tf.image.random_saturation(x, 0.2, 1.8)
            x = tf.image.random_hue(x, 0.2)
            return tf.clip_by_value(x, 0.0, 1.0)

        return tf.cond(p < 0.8, apply, lambda: image)


    ########################################################
    # Random Grayscale
    ########################################################

    def random_grayscale(self, image):

        p = tf.random.uniform(())

        return tf.cond(p < 0.2, lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image)), lambda: image)



    # Gaussian Blur
    # def gaussian_blur(self, image):
    #     return image


    ########################################################
    # Full Pipeline
    ########################################################

    def __call__(self, image):

        image = self.random_resized_crop(image)
        image = self.horizontal_flip(image)
        image = self.color_jitter(image)
        image = self.random_grayscale(image)
        #image = self.gaussian_blur(image)
        return image


#########################################################
# Two Views
#########################################################

class SimCLRTransform:

    def __init__(self):

        self.augment = SimCLRAugmentation()


    def __call__(self, image):

        view1 = self.augment(image)
        view2 = self.augment(image)
        return view1, view2