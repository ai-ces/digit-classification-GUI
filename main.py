from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap # icon and load image
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QWidget
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # load image

import cv2 #open-cv use for image resize

import keras # keras tensorflow
from keras.datasets import mnist
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

#%% preprocess

(x_train, y_train),(x_test, y_test) = mnist.load_data()

plt.figure()
plt.imshow(x_train[1], cmap = "gray")
plt.axis("off")
plt.grid(False)

img_rows = 28
img_cols = 28
x_train = x_train.reshape( x_train.shape[0], img_rows,img_cols,1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows,img_cols,1)

# normalization

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# %% CNN

model_list = []
score_list = []

batch_size = 256
epochs = 5

filter_numbers = np.array([[16,32,64],[8,16,32]])

for i in range(2):
    print(filter_numbers[i])
    model = Sequential()
    model.add(Conv2D(filter_numbers[i,0], kernel_size = (3,3), activation = "relu", input_shape = input_shape))
    model.add(Conv2D(filter_numbers[i,1], kernel_size = (3,3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(filter_numbers[i,2], activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation = "softmax"))

    model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(),  metrics = ["accuracy"])

    history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose = 0)
    print("Model {} Test Loss {}".format(i+1, score[0]))
    print("Model {} Test Accuracy {}".format(i+1, score[1]))
    model_list.append(model)
    score_list.append(score)

    model.save("model"+str(i+1)+".h5")

# %%

    model1 = load_model("model1.h5")
    model1 = load_model("model2.h5")

# %% GUI

class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        #main window
        self.width = 1080
        self.height = 640

        self.setWindowTitle("Digit Classification")
        self.setGeometry(50,100,self.width, self.height)
        
        self.tabWidget()
        self.widgets()
        self.layouts()
        self.show()
    
    def tabWidget(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab1 = QWidget()
        self.tab2 = QWidget()

        self.tabs.addTab(self.tab1, "Classification")
        self.tabs.addTab(self.tab2, "Parameters")

    def widgets(self):
        #tab1 left
        self.drawCanvas = QPushButton("Draw Canvas")
        self.drawCanvas.clicked.connect(self.drawCanvasFunction)

        self.openCanvas = QPushButton("Open Canvas")
        self.openCanvas.clicked.connect(self.openCanvasFunction)

        self.inputImage = QLabel(self)
        self.inputImage.setPixmap(QPixmap("input.png"))

        self.searchText = QLabel("Real Number: ")

        self.searchEntry = QLineEdit()
        self.searchEntry.setPlaceholderText("Which number do yo write?")


        # tab1 left middle

        self.methodSelection = QComboBox(self)
        self.methodSelection.addItems(["model1","model2"])

        self.noiseText = QLabel("Add Noise: % " + "0")
        self.noiseSlider = QSlider(Qt.Horizontal)
        self.noiseSlider.setMinimum(0)
        self.noiseSlider.setMaximum(100)
        self.noiseSlider.setTickPosition(QSlider.TicksBelow)
        self.noiseSlider.setTickInterval(1)
        self.noiseSlider.valueChanged.connect(self.noiseSliderFunction)

        self.remember = QCheckBox("Save Result", self)

        self.predict = QPushButton("Predict")
        self.predict.clicked.connect(self.predictionFunction)

        # tab1 right middle

        self.outputImage = QLabel(self)
        self.outputImage.setPixmap(QPixmap("bluee.png"))

        self.outputLabel = QLabel("", self)
        self.outputLabel.setAlignment(Qt.AlignCenter)

        # tab 1 right

        self.resultTable = QTableWidget()
        self.resultTable.setColumnCount(2)
        self.resultTable.setRowCount(10)
        self.resultTable.setHorizontalHeaderItem(0, QTableWidgetItem("Label(Class)"))
        self.resultTable.setHorizontalHeaderItem(1, QTableWidgetItem("Probability"))

        # tab2 method1

        self.parameter_list1 = QListWidget(self)
        self.parameter_list1.addItems(["batch_size = 256", "epochs = 5", "img_rows = 28", "img_cols = 28", "Filter # = [16,32,64]", "Activation Function = Relu", "loss = categorucall cross entropy", "optimizer = Adadelta", "metrics = accuracy"])

        # tab2 method2

        self.parameter_list2 = QListWidget(self)
        self.parameter_list2.addItems(["batch_size = 256", "epochs = 5", "img_rows = 28", "img_cols = 28", "Filter # = [8,16,32]", "Activation Function = Relu", "loss = categorucall cross entropy", "optimizer = Adadelta", "metrics = accuracy"])
        


    def drawCanvasFunction(self):
        pass
    def openCanvasFunction(self):
        pass
    def predictionFunction(self):
        pass
    
    def noiseSliderFunction(self):
        val = self.noiseSlider.value()
        self.noiseText.setText("Add Noise: % " + str(val))

    def layouts(self):

        self.mainLayout = QHBoxLayout()
        self.leftLayout = QFormLayout()
        self.leftMiddleLayout = QFormLayout()
        self.rightMiddleLayout = QFormLayout()
        self.rightLayout = QFormLayout()

        # Left

        self.leftLayoutGroupbox = QGroupBox("Input Image")
        self.leftLayout.addRow(self.drawCanvas)
        self.leftLayout.addRow(self.openCanvas)
        self.leftLayout.addRow(self.inputImage)
        self.leftLayout.addRow(self.searchText)
        self.leftLayout.addRow(self.searchText)
        self.leftLayoutGroupbox.setLayout(self.leftLayout)

        # left middle

        
        self.leftMiddleLayoutGroupbox = QGroupBox("Settings")
        self.leftMiddleLayout.addRow(self.methodSelection)
        self.leftMiddleLayout.addRow(self.noiseText)
        self.leftMiddleLayout.addRow(self.noiseSlider)
        self.leftMiddleLayout.addRow(self.remember)
        self.leftMiddleLayout.addRow(self.predict)
        self.leftMiddleLayoutGroupbox.setLayout(self.leftMiddleLayout)

        # right middle

        self.rightMiddleLayoutGroupBox = QGroupBox("Output")
        self.leftMiddleLayout.addRow(self.outputImage)
        self.leftMiddleLayout.addRow(self.outputLabel)
        self.rightMiddleLayoutGroupBox.setLayout(self.rightMiddleLayout)

        # right

        self.rightLayoutGroupBox = QGroupBox("Result")
        self.rightLayout.addRow(self.resultTable)
        self.rightLayoutGroupBox.setLayout(self.rightLayout)

        # tab1 main Layout

        self.mainLayout.addWidget(self.leftLayoutGroupbox)
        self.mainLayout.addWidget(self.leftMiddleLayoutGroupbox)
        self.mainLayout.addWidget(self.rightMiddleLayoutGroupBox)
        self.mainLayout.addWidget(self.rightLayoutGroupBox)
        self.tab1.setLayout(self.mainLayout)

        # tab 2 Layout

        self.tab2Layout = QHBoxLayout()
        self.tab2Method1Layout = QFormLayout()
        self.tab2Method2Layout = QFormLayout()

        # tab2 Method1 Layout

        self.tab2Method1LayoutGroupBox = QGroupBox("Method1")
        self.tab2Method1Layout.addRow(self.parameter_list1)
        self.tab2Method1LayoutGroupBox.setLayout(self.tab2Method1Layout)

        # tab2 Method2 Layout

        self.tab2Method2LayoutGroupbox = QGroupBox("Method2")
        self.tab2Method2Layout.addRow(self.parameter_list2)
        self.tab2Method2LayoutGroupbox.setLayout(self.tab2Method2Layout)

        # tab2 main Layout

        self.tab2Layout.addWidget(self.tab2Method1LayoutGroupBox, 50)
        self.tab2Layout.addWidget(self.tab2Method2LayoutGroupbox, 50)
        self.tab2.setLayout(self.tab2Layout)



w = Window()