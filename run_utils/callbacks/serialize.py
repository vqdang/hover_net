
import matplotlib
import numpy as np
# import seaborn as sn
from matplotlib import pyplot as plt

# * syn where to set this
# must use 'Agg' to plot out onto image
# matplotlib.use('Agg')


####
def fig2data(fig):
    """
    Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    Args:
        fig: a matplotlib figure
    
    Return: a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode.
    # Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


####
class _Scalar(object):
    @staticmethod
    def to_console(value):
        return '%0.5f' % value

    @staticmethod
    def to_json(value):
        return value

    @staticmethod
    def to_tensorboard(value):
        return 'scalar', value

####
class _ConfusionMatrix(object):
    @staticmethod
    def to_console(value):
        value = pd.DataFrame(value)
        value.index.name = 'True'
        value.columns.name = 'Pred'
        formatted_value = value.to_string()
        return '\n' + formatted_value

    @staticmethod
    def to_json(value):
        value = pd.DataFrame(value)
        value.index.name = 'True'
        value.columns.name = 'Pred'
        value = value.unstack().rename('value').reset_index()
        value = pd.Series({'conf_mat': value})
        formatted_value = value.to_json(orient='records')
        return formatted_value

    @staticmethod
    def to_tensorboard(value):
        # assert matplotlib.get_backend() == 'Agg'
        value = pd.DataFrame(value)
        value.index.name = 'True'
        value.columns.name = 'Pred'
        fig = plt.figure(figsize=(10, 10))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(value, annot=True, annot_kws={"size": 16})  # font size
        img = np.transpose(fig2data(fig), axes=[2, 0, 1])  # HWC => CHW
        plt.close()
        return 'image', img

####
class _Image(object):
    @staticmethod
    def to_console(value):
        # TODO: add warn for not possible or sthg here
        return None

    @staticmethod
    def to_json(value):
        # TODO: add warn for not possible or sthg here
        return None

    @staticmethod
    def to_tensorboard(value):
        # TODO: add method
        return 'image', value


__converter_dict = {
    'scalar': _Scalar,
    'conf_mat': _ConfusionMatrix,
    'image': _Image
}


####
def serialize(value, input_format, output_format):
    converter = __converter_dict[input_format]
    if output_format == 'console':
        return converter.to_console(value)
    elif output_format == 'json':
        return converter.to_json(value)
    elif output_format == 'tensorboard':
        return converter.to_tensorboard(value)
    else:
        assert False, 'Unknown format'
