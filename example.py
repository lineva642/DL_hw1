from network import Network
from dataset import Data
from trainer import Trainer, get_data_from_datasets, predict_proba_on_mesh_tensor
from visualization_utils import make_meshgrid,  plot_predictions
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

#задаем слои сети
list_of_layers = [(2, 5), (5, 3), (3, 2)]
#создаем сеть с заданной конфигурацией
nn_model = Network(list_of_layers)
#вызываем класс для тренировки сети
trainer = Trainer(nn_model, lr=0.1)

#задаем тренировочные и тестовые данные
train_data = Data(n_samples = 5000, shuffle = True, noise=0.3, random_state=0, factor = 0.5)
test_data = Data(n_samples = 500, shuffle = True, noise=0.3, random_state=2, factor = 0.5)

train_dataloader = DataLoader(train_data, batch_size=50, shuffle = False)
test_dataloader = DataLoader(test_data, batch_size=50, shuffle = False)

#тренируем сеть
trainer.fit(train_dataloader, n_epochs = 100)
#делаем предсказание на тестовых данных
test_predictions_proba = trainer.predict_proba(test_dataloader)

#########################################ДОБАВЛЕННАЯ_ЧАСТЬ#################################################
#оценим полученный результат
y_pred = [0 if test_predictions_proba[i][0] > test_predictions_proba[i][1] else 1 for i in range(test_predictions_proba.shape[0])]
print("Accuracy: ", accuracy_score(test_data.y, y_pred))
###########################################################################################################

#визуализируем результат и сохраняем в отдельный файл
X_train, X_test, y_train, y_test = get_data_from_datasets(train_data, test_data)

xx, yy = make_meshgrid(X_train, X_test)

Z = predict_proba_on_mesh_tensor(trainer, xx, yy)
plot_predictions(xx, yy, Z, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test,
                 plot_name = 'prediction.png')
