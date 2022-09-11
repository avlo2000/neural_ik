from data.generators import RandomGen
import tensorflow as tf

from data.robots import arm7dof


def main():
    robot = arm7dof()
    gen_valid = RandomGen(robot=robot, batch_size=1, n=1000)
    model = tf.keras.models.load_model(r'./models/dnn_solver.hdf5')
    res = model.predict(gen_valid, verbose=1)
    print(res)


if __name__ == '__main__':
    main()
