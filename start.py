
from json3d.resize_photo_lables import make_csv
from train import train
from test import test_f
from json3d.test_on_plot import test_on_plot

make_csv()
test_on_plot()
train()
test_f(r'C:\Users\PC\PycharmProjects\3D_model_car_object_detection\json3d\norm_img')
