import tensorcircuit as tc
from matplotlib import style
import tensorflow as tf
import numpy as np
import random
import time
from functools import partial
style.use('bmh')

batch_size = 100
nqubits = 8
path_len = 5
population_size_1 = 10
population_size_2 = 10

K = tc.set_backend("tensorflow")
tc.set_dtype("complex64")

path = r'/packages_8/the_repository_of_quantum_combination_logic_gates.npz'
Column_set_all = np.load(path)
Column_set = Column_set_all[Column_set_all.files[0]]

path = r'/packages_8/the_connectivity_repository_of_quantum_combination_logic_gates.npz'
direct_matrix_all = np.load(path)
direct_matrix = direct_matrix_all[direct_matrix_all.files[0]]

shape = direct_matrix.shape
const1 = shape[1]-1
const2 = shape[0]-1
mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
x_train_q, x_test_q = X_train / 255.0, X_test / 255.0

y_train_q = tf.one_hot(Y_train, depth=10)
y_test_q = tf.one_hot(Y_test, depth=10)
n_sample = 600
test_sample = 100
x_train, x_test, y_train, y_test = x_train_q[:n_sample], x_test_q[:test_sample], y_train_q[:n_sample], y_test_q[:test_sample]
def add_number(encode_number, k):
    encode_number_copy = []
    encode_number_copy.append(encode_number)
    for i in range(k):
        temp = encode_number_copy[len(encode_number_copy)-1]
        temp1 = direct_matrix[:, temp]
        temp2 = temp1[random.randint(0, const2)]
        while temp2 == 0:
            temp2 = temp1[random.randint(0, const2)]
        encode_number_copy.append(temp2)
    return encode_number_copy
def generate_initial_path(path_length):
    initial_code_1 = np.random.randint(0, const1, 1)
    initial_code = initial_code_1[0]
    a = add_number(initial_code, path_length-1)
    return a
def generate_next_path(front_path,next_path_num):
    initial_code = front_path[len(front_path)-1]
    a = add_number(initial_code, next_path_num)
    a.pop(0)
    return a
def generate_initial_population(path_len, populations):
    population = [[0 for j in range(path_len)] for i in range(populations)]
    for i in range(populations):
        temp = generate_initial_path(path_len)
        population[i] = temp
    return population
def generate_next_population(front_path_list, path_next_len, populations):
    population = [[0 for j in range(len(front_path_list)+path_next_len)] for i in range(populations)]
    for i in range(populations):
        temp = generate_next_path(front_path_list, path_next_len)
        population[i][0:len(front_path_list)] = front_path_list
        population[i][len(front_path_list):len(front_path_list)+path_next_len] = temp
    return population

population_initial = generate_initial_population(path_len, population_size_1)

def circuit_get_out(x, weights,circuit_path_number):
    nqubits = 8
    circuit_number = circuit_path_number
    c_all = tc.Circuit(nqubits, inputs=x)
    w_num = 0
    for p in range(nqubits):
        c_all.H(p)
    for i in range(len(circuit_number)):
        index = Column_set[:, circuit_number[i]]
        for j in (0,2,4,6):
            m = j+1
            if index[j]>0:
                c_all.CRX(index[j] - 1, index[m] - 1, theta=weights[0][w_num])
                w_num = w_num+1
        for k in range(8, 16):
            if index[k] > 0:
                w_x1_num = w_num
                w_z_num = w_x1_num+1
                w_x2_num = w_z_num+1
                c_all.RX(index[k] - 1, theta=weights[0][w_x1_num])
                c_all.RZ(index[k] - 1, theta=weights[0][w_z_num])
                c_all.RX(index[k] - 1, theta=weights[0][w_x2_num])
                w_num = w_x2_num+1
    out = K.stack([K.real(c_all.expectation([tc.gates.x(), [i]])) for i in range(nqubits)]
                  + [K.real(c_all.expectation([tc.gates.y(), [i]])) for i in range(nqubits)]
                  + [K.real(c_all.expectation([tc.gates.z(), [i]])) for i in range(nqubits)]
                  )
    return out

def train(epochs, circuit_path):
    stat_time = time.time()
    circuit_number = population_initial[circuit_path]
    w_all_num = 0
    for i in range(len(circuit_number)):
        index = Column_set[:, circuit_number[i]]
        w_i_path_num_cr = 0
        w_i_path_num_r = 0
        for i in (0, 2, 4, 6):
            if index[i] > 0:
                w_i_path_num_cr = w_i_path_num_cr + 1
        for i in range(nqubits, 2 * nqubits):
            if index[i] > 0:
                w_i_path_num_r = w_i_path_num_r + 1
        w_all_num = w_all_num + w_i_path_num_cr + w_i_path_num_r * 3

    qml_layer = tc.keras.QuantumLayer(partial(circuit_get_out, circuit_path_number=circuit_number), weights_shape=[(1, w_all_num)])
    inputs = tf.keras.Input(shape=(28, 28, 1))
    conve1 = tf.keras.layers.Conv2D(filters=10, strides=(2, 2), kernel_size=(4, 4), activation='relu')(inputs)
    conve2 = tf.keras.layers.Conv2D(filters=16, strides=(3, 3), kernel_size=(4, 4), activation='relu')(conve1)
    flatt = tf.keras.layers.Flatten()(conve2)
    qml = qml_layer(flatt)
    output = tf.keras.layers.Dense(10, activation="softmax")(qml)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.051),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    History = model.fit(x_train, y_train, batch_size=100, epochs=epochs, validation_data=[x_test, y_test])
    train_lossy = History.history['loss']
    train_accy = History.history['categorical_accuracy']
    val_lossy = History.history['val_loss']
    val_accy = History.history['val_categorical_accuracy']
    model_param = model.count_params()
    end_time = time.time()
    print(f"the running time is: {end_time - stat_time} s")

    return train_lossy, train_accy, val_lossy, val_accy, model_param
def evolution_next_path():
    max_acc_path = []
    max_loss_path = []
    para_path = []
    for i in range(population_size_1):
        loss, acc, val_loss, val_acc, model_par = train(10, i)
        max_acc = max(acc)
        max_acc_index = acc.index(max(acc))
        max_acc_path.append(max_acc)
        para_path.append(model_par)
        max_loss_path.append(loss[max_acc_index])
    print(max_acc_path)
    print(max_loss_path)
    print(para_path)
    max_path_num = max_acc_path.index(max(max_acc_path))
    fro_path = population_initial[max_path_num]
    next_path = generate_next_population(fro_path, 2, population_size_2)
    print("1-th_Acc：", max_acc_path[max_path_num])
    print("1_th_params：", para_path[max_path_num])
    print("1_th_optimal_ELCS：", fro_path)
    return next_path
class evolution_next_maxacc_path:
    def __init__(self, current_mult_path, epoch_num):
        self.current_mult_path = current_mult_path
        self.epoch_num = epoch_num

    def circuit_get_out_class(self, x, weights, circuit_path_number):
        nqubits = 8
        circuit_number = circuit_path_number
        c_all = tc.Circuit(nqubits, inputs=x)
        w_num = 0
        for p in range(nqubits):
            c_all.H(p)
        for i in range(len(circuit_number)):
            index = Column_set[:, circuit_number[i]]
            for j in (0, 2, 4, 6):
                m = j + 1
                if index[j] > 0:
                    c_all.CRX(index[j] - 1, index[m] - 1, theta=weights[0][w_num])
                    w_num = w_num + 1
            for k in range(8, 16):
                if index[k] > 0:
                    w_x1_num = w_num
                    w_z_num = w_x1_num + 1
                    w_x2_num = w_z_num + 1
                    c_all.RX(index[k] - 1, theta=weights[0][w_x1_num])
                    c_all.RZ(index[k] - 1, theta=weights[0][w_z_num])
                    c_all.RX(index[k] - 1, theta=weights[0][w_x2_num])
                    w_num = w_x2_num + 1
        out = K.stack([K.real(c_all.expectation([tc.gates.x(), [i]]))for i in range(nqubits)]
                      + [K.real(c_all.expectation([tc.gates.y(), [i]]))for i in range(nqubits)]
                      + [K.real(c_all.expectation([tc.gates.z(), [i]]))for i in range(nqubits)])
        return out

    def train_class(self, epochs, circuit_path):
        circuit_number = self.current_mult_path[circuit_path]
        w_all_num_class = 0
        for i in range(len(circuit_number)):
            index = Column_set[:, circuit_number[i]]
            w_i_path_num_cr = 0
            w_i_path_num_r = 0
            for i in (0, 2, 4, 6):
                if index[i] > 0:
                    w_i_path_num_cr = w_i_path_num_cr + 1
            for i in range(nqubits, 2 * nqubits):
                if index[i] > 0:
                    w_i_path_num_r = w_i_path_num_r + 1
            w_all_num_class = w_all_num_class + w_i_path_num_cr + w_i_path_num_r * 3

        qml_layer = tc.keras.QuantumLayer(partial(self.circuit_get_out_class,
                                                  circuit_path_number=circuit_number), weights_shape=[(1, w_all_num_class)])
        inputs = tf.keras.Input(shape=(28, 28, 1))
        conve1 = tf.keras.layers.Conv2D(filters=10, strides=(2, 2), kernel_size=(4, 4), activation='relu')(inputs)
        conve2 = tf.keras.layers.Conv2D(filters=16, strides=(3, 3), kernel_size=(4, 4), activation='relu')(conve1)
        flatt = tf.keras.layers.Flatten()(conve2)
        qml = qml_layer(flatt)
        output = tf.keras.layers.Dense(10, activation="softmax")(qml)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.051),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )
        History = model.fit(x_train, y_train, batch_size=100, epochs=epochs, validation_data=[x_test, y_test], verbose=0)
        train_lossy = History.history['loss']
        train_accy = History.history['categorical_accuracy']
        val_lossy = History.history['val_loss']
        val_accy = History.history['val_categorical_accuracy']
        model_param = model.count_params()
        return train_lossy, train_accy, val_lossy, val_accy, model_param

    def evolution_class(self):
        max_acc_path = []
        max_loss_path = []
        para_path = []
        for i in range(population_size_2):
            loss, acc, val_loss, val_acc, model_par = evolution_next_maxacc_path.train_class(self, self.epoch_num, i)
            max_acc = max(acc)
            max_acc_index = acc.index(max(acc))
            max_acc_path.append(max_acc)
            para_path.append(model_par)
            max_loss_path.append(loss[max_acc_index])
        max_path_num = max_acc_path.index(max(max_acc_path))
        fro_path = self.current_mult_path[max_path_num]
        next_path = generate_next_population(fro_path, 2, population_size_2)
        return fro_path, next_path, max_acc_path, max_loss_path, para_path, max(max_acc_path), max_loss_path[max_path_num], para_path[max_path_num]

def main():
    stat_all_time = time.time()
    a = evolution_next_path()
    epoch_num = list(range(15, 100, 5))
    evo_net_path = evolution_next_maxacc_path(a, epoch_num[0])
    for i in range(100):
        optimal_path, optimal_mult_path, acc_path, loss_path, para_path, acc, loss, para_num = evo_net_path.evolution_class()
        evo_net_path.current_mult_path = optimal_mult_path
        evo_net_path.epoch_num = epoch_num[i + 1]
        acc_round = round(acc, 4)
        print(i + 2, "acc_list", acc_path)
        print(i + 2, "loss_list", loss_path)
        print(i + 2, "para_list", para_path)

        print(i + 2, "acc", acc_round)
        print(i + 2, "loss", loss)
        print(i + 2, "sum(para)", para_num)
        print(i + 2, "optimal_path", optimal_path)
        print(i + 2, "path_depth", len(optimal_mult_path[0]) - 2)
        if i > 20 or acc_round > 0.99:
             break
    end_all_time = time.time()
    print(f"the running time is: {end_all_time - stat_all_time} s")

if __name__ == '__main__':
     main()
