import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

# * syn where to set this
# must use 'Agg' to plot out onto image
matplotlib.use("Agg")

####
def fig2data(fig, dpi=180):
    """Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it.
    
    Args:
        fig: a matplotlib figure
    
    Return: a numpy 3D array of RGBA values

    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


####
class _Scalar(object):
    @staticmethod
    def to_console(value):
        return "%0.5f" % value

    @staticmethod
    def to_json(value):
        return value

    @staticmethod
    def to_tensorboard(value):
        return "scalar", value


####
class _ConfusionMatrix(object):
    @staticmethod
    def to_console(value):
        value = pd.DataFrame(value)
        value.index.name = "True"
        value.columns.name = "Pred"
        formatted_value = value.to_string()
        return "\n" + formatted_value

    @staticmethod
    def to_json(value):
        value = pd.DataFrame(value)
        value.index.name = "True"
        value.columns.name = "Pred"
        value = value.unstack().rename("value").reset_index()
        value = pd.Series({"conf_mat": value})
        formatted_value = value.to_json(orient="records")
        return formatted_value

    @staticmethod
    def to_tensorboard(value):
        def plot_confusion_matrix(
            cm, target_names, title="Confusion matrix", cmap=None, normalize=False
        ):
            """given a sklearn confusion matrix (cm), make a nice plot.

            Args:
                cm:           confusion matrix from sklearn.metrics.confusion_matrix

                target_names: given classification classes such as [0, 1, 2]
                            the class names, for example: ['high', 'medium', 'low']

                title:        the text to display at the top of the matrix

                cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                            see http://matplotlib.org/examples/color/colormaps_reference.html
                            plt.get_cmap('jet') or plt.cm.Blues

                normalize:    If False, plot the raw numbers
                            If True, plot the proportions

            Usage
            -----
            plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                    # sklearn.metrics.confusion_matrix
                                normalize    = True,                # show proportions
                                target_names = y_labels_vals,       # list of names of the classes
                                title        = best_estimator_name) # title of graph

            Citiation
            ---------
            http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

            """
            import matplotlib.pyplot as plt
            import numpy as np
            import itertools

            accuracy = np.trace(cm) / np.sum(cm).astype("float")
            misclass = 1 - accuracy

            if cmap is None:
                cmap = plt.get_cmap("Blues")

            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation="nearest", cmap=cmap)
            plt.title(title)
            plt.colorbar()

            if target_names is not None:
                tick_marks = np.arange(len(target_names))
                plt.xticks(tick_marks, target_names, rotation=45)
                plt.yticks(tick_marks, target_names)

            if normalize:
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            thresh = cm.max() / 1.5 if normalize else cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(
                        j,
                        i,
                        "{:0.4f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )
                else:
                    plt.text(
                        j,
                        i,
                        "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )

            plt.tight_layout()
            plt.ylabel("True label")
            plt.xlabel(
                "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(
                    accuracy, misclass
                )
            )

        plot_confusion_matrix(value, ["0", "1"])
        img = fig2data(plt.gcf())
        plt.close()
        return "image", img


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
        return "image", value


__converter_dict = {"scalar": _Scalar, "conf_mat": _ConfusionMatrix, "image": _Image}


####
def serialize(value, input_format, output_format):
    converter = __converter_dict[input_format]
    if output_format == "console":
        return converter.to_console(value)
    elif output_format == "json":
        return converter.to_json(value)
    elif output_format == "tensorboard":
        return converter.to_tensorboard(value)
    else:
        assert False, "Unknown format"
