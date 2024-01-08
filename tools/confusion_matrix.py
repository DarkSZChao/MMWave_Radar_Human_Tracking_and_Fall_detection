import matplotlib.pyplot as plt
import numpy as np

matrix = np.array([[216, 16, 0],
                   [6, 66, 2],
                   [0, 1, 56]])

sub_sum = matrix.sum(axis=1)

confusion_matrix = (matrix.T / sub_sum).T

classes = ['Standing', 'Sitting', 'Fall Detection']

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

for i in range(len(classes)):
    for j in range(len(classes)):
        cell_value = confusion_matrix[i, j]
        cell_text = "{:.1%}".format(cell_value / np.sum(confusion_matrix[i, :]))
        plt.text(j, i, cell_text, horizontalalignment='center', color='white' if cell_value > np.max(confusion_matrix) / 2 else 'black')

plt.tight_layout()
plt.show()
