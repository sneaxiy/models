import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
import paddle.fluid.unique_name as unique_name
import models.se_resnext_using_py_reader
import reader
import argparse
import functools
from models.learning_rate import cosine_decay
from utility import add_arguments, print_arguments
import math
import threading

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,   256,                  "Minibatch size.")
add_arg('use_gpu',          bool,  True,                 "Whether to use GPU or not.")
add_arg('total_images',     int,   1281167,              "Training image number.")
add_arg('num_epochs',       int,   120,                  "number of epochs.")
add_arg('class_dim',        int,   1000,                 "Class number.")
add_arg('image_shape',      str,   "3,224,224",          "input image size")
add_arg('with_mem_opt',     bool,  True,                 "Whether to use memory optimization or not.")
add_arg('lr',               float, 0.1,                  "set learning rate.")
add_arg('lr_strategy',      str,   "piecewise_decay",    "Set the learning rate decay strategy.")
add_arg('model',            str,   "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('queue_capacity',   int,   64,                   "Set the capacity of data feeding queue.")
# yapf: enable

model_list = [
    m for m in dir(models.se_resnext_using_py_reader) if "__" not in m
]


def optimizer_setting(params):
    ls = params["learning_strategy"]

    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]

        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        bd = [step * e for e in ls["epochs"]]
        base_lr = params["lr"]
        lr = []
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    elif ls["name"] == "cosine_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]

        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        lr = params["lr"]
        num_epochs = params["num_epochs"]

        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    else:
        lr = params["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    return optimizer


def get_model(args, for_test=False):
    return models.se_resnext_using_py_reader.__dict__[args.model](
        feed_queue_capacity=args.queue_capacity,
        for_test=for_test).net(class_dim=args.class_dim)


def as_tensor(np_or_tensor, place=None):
    if isinstance(np_or_tensor, fluid.LoDTensor):
        return np_or_tensor

    tensor = fluid.LoDTensor()
    tensor.set(np_or_tensor, place if place is not None else fluid.CPUPlace())
    return tensor


def as_numpy(tensor_or_np):
    return tensor_or_np if isinstance(tensor_or_np,
                                      np.ndarray) else np.array(tensor_or_np)


def feed_data(feed_queue, reader, sync_signal, epoch_num, image_shape):
    image_shape = [-1] + image_shape
    for epoch_id in range(epoch_num):
        data_generator = reader()
        while True:
            next_data = next(data_generator, None)
            if next_data is None:
                sync_signal.acquire()
                feed_queue.close()
                while feed_queue.is_closed():
                    sync_signal.wait()
                sync_signal.release()
                break

            image = [next_data[i][0] for i in range(len(next_data))]
            label = [next_data[i][1] for i in range(len(next_data))]
            image = np.stack(image, axis=0)
            label = np.array(label).reshape([-1, 1])

            tensors = fluid.LoDTensorArray()
            tensors.append(as_tensor(image.reshape(image_shape)))
            tensors.append(as_tensor(label.reshape([-1, 1])))
            feed_queue.push(tensors)

    feed_queue.close()


def train(args):
    # parameters from arguments
    class_dim = args.class_dim
    model_name = args.model
    with_memory_optimization = args.with_mem_opt
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    params = models.se_resnext_using_py_reader.train_parameters
    params["total_images"] = args.total_images
    params["lr"] = args.lr
    params["num_epochs"] = args.num_epochs
    params["learning_strategy"]["batch_size"] = args.batch_size
    params["learning_strategy"]["name"] = args.lr_strategy
    params["input_size"] = image_shape

    # model definition
    train_main_program, train_startup_program, avg_cost, acc_top1, acc_top5, \
            train_py_reader, train_queue = get_model(args, for_test=False)

    test_main_program, test_startup_program, _, _, _, \
            test_py_reader, test_queue = get_model(args, for_test=True)

    # initialize optimizer
    with fluid.program_guard(train_main_program, train_startup_program):
        optimizer = optimizer_setting(params)
        opts = optimizer.minimize(avg_cost)

    if with_memory_optimization:
        fluid.memory_optimize(train_main_program)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place=place)
    exe.run(train_startup_program)

    train_batch_size = args.batch_size
    test_batch_size = 16
    epoch_num = params["num_epochs"]

    train_reader = paddle.batch(reader.train(), batch_size=train_batch_size)
    test_reader = paddle.batch(reader.val(), batch_size=test_batch_size)

    train_sync_signal = threading.Condition()
    train_feed_thread = threading.Thread(
        target=feed_data,
        args=(train_queue, train_reader, train_sync_signal, epoch_num,
              image_shape))

    train_exe = fluid.ParallelExecutor(
        main_program=train_main_program, use_cuda=True, loss_name=avg_cost.name)
    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]
    train_feed_thread.start()
    for pass_id in range(epoch_num):
        batch_id = 0
        while True:
            try:
                t1 = time.time()
                loss, acc1, acc5 = train_exe.run(fetch_list=fetch_list)
                t2 = time.time()
                period = t2 - t1
                loss = np.mean(as_numpy(loss))
                acc1 = np.mean(as_numpy(acc1))
                acc5 = np.mean(as_numpy(acc5))
                if batch_id % 1 == 0:
                    print(
                        "Pass {0}, trainbatch {1}, loss {2}, acc1 {3}, acc5 {4} time {5}"
                        .format(pass_id, batch_id, loss, acc1, acc5, "%2.2f sec"
                                % period))
                    sys.stdout.flush()
                batch_id += 1
            except fluid.core.EOFException as ex:
                print(ex)
                train_sync_signal.acquire()
                train_py_reader.reset()
                train_sync_signal.notify()
                train_sync_signal.release()
                break

    exe.run(test_startup_program)

    test_epoch_num = 1
    test_sync_signal = threading.Condition()
    test_feed_thread = threading.Thread(
        target=feed_data,
        args=(test_queue, test_reader, test_sync_signal, 1, image_shape))
    test_feed_thread.start()
    batch_id = 0
    while True:
        try:
            t1 = time.time()
            loss, acc1, acc5 = exe.run(test_main_program, fetch_list=fetch_list)
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(loss)
            acc1 = np.mean(acc1)
            acc5 = np.mean(acc5)
            if batch_id % 10 == 0:
                print("Testbatch {0},loss {1}, acc1 {2},acc5 {3},time {4}"
                      .format(batch_id, loss, acc1, acc5, "%2.2f sec" % period))
                sys.stdout.flush()
            batch_id += 1
        except fluid.core.EOFException as ex:
            test_sync_signal.acquire()
            test_py_reader.reset()
            test_sync_signal.notify()
            test_sync_signal.release()
            break


def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args)


if __name__ == '__main__':
    main()
