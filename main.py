from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf
import numpy, time, threading
import os

from model import SmallMLP, MediumMLP, BigMLP

# Perform training
def training_process(train_operations, x_train, y_train):
    print ('---------------- Processo de Treinamento ----------------')
    
    train_threads = []
    for train_operation in train_operations:
        train_threads.append(threading.Thread(target = train_operation, 
                                              args   = (x_train, y_train)))
    
    print ('Iniciando Threads')
    start_time = time.time()
    
    for train_thread in train_threads:
        train_thread.start()
    for train_thread in train_threads:
        train_thread.join()
    
    minutes = (time.time() - start_time) / 60
    seconds = (time.time() - start_time) % 60
    print('Processo concluído!\n'
          'Todas as Threads foram finalizadas!\n'
          'Tempo Gasto: %.0f minuto(s) e %.2f segundo(s)' % (minutes, seconds))

    print ('\nExecuntando em Única Thread')
    start_time = time.time()

    train_operations[0](x_train, y_train)
    train_operations[1](x_train, y_train)
    train_operations[2](x_train, y_train)
    
    minutes = (time.time() - start_time) / 60
    seconds = (time.time() - start_time) % 60
    print('Processo concluído!\n'
          'Tempo Gasto: %.0f minuto(s) e %.2f segundos' % (minutes, seconds))

# Perform testing
def test_process(test_operations, x_test, y_test):
    print ('---------------- Processo de Teste ----------------')
    
    test_threads = []
    for test_operation in test_operations:
        test_threads.append(threading.Thread(target  = test_operation, 
                                              args   = (x_test, y_test)))

    print ('Iniciando Threads')
    start_time = time.time()
    
    for test_thread in test_threads:
        test_thread.start()
    for test_thread in test_threads:
        test_thread.join()

    minutes = (time.time() - start_time) / 60
    seconds = (time.time() - start_time) % 60
    print('Processo concluído!\n'
          'Todas as Threads foram finalizadas!\n'
          'Tempo Gasto: %.0f minuto(s) e %.2f segundo(s)' % (minutes, seconds))
    
    print ('\nExecuntando em Única Thread')
    start_time = time.time()
    
    test_operations[0](x_test, y_test)
    test_operations[1](x_test, y_test)
    test_operations[2](x_test, y_test)

    minutes = (time.time() - start_time) / 60
    seconds = (time.time() - start_time) % 60
    print('Processo concluído!\n'
          'Tempo Gasto: %.0f minuto(s) e %.2f segundo(s)' % (minutes, seconds))

# Load datasets and generate train and test
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test  = x_test.reshape(10000,  784)

    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test,  10)

    return x_train, y_train, x_test, y_test

# Main function
def main():
    numpy.random.seed(1)
    x_train, y_train, x_test, y_test = load_data()

    small_MLP  = SmallMLP()
    medium_MLP = MediumMLP()
    big_MLP    = BigMLP()

    small_MLP_train,  small_MLP_test  = small_MLP.train_and_test_operations()
    medium_MLP_train, medium_MLP_test = medium_MLP.train_and_test_operations()
    big_MLP_train,    big_MLP_test    = big_MLP.train_and_test_operations()
    
    train_operations  = [small_MLP_train, medium_MLP_train, big_MLP_train]    
    test_operations   = [small_MLP_test,  medium_MLP_test,  big_MLP_test]
    
    training_process(train_operations, x_train, y_train)
    test_process(test_operations, x_test, y_test)
    
    print("Experimento concluído!")

if __name__ == '__main__':    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
    
