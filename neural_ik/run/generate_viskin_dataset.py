import os
from data import generators, robots
from data.data_io import paths_to_dataset, generate_to_file


def main():
    robot = robots.arm6dof()
    size = 10_000_000
    train_test_split = 0.8
    path_to_train, path_to_test = paths_to_dataset('fk_ds_0_0_3')

    gen_train = generators.RandomGen(robot=robot, batch_size=32, n=int(size * train_test_split))
    gen_test = generators.RandomGen(robot=robot, batch_size=32, n=int(size * (1 - train_test_split)))

    generate_to_file(gen_train, path_to_train)
    generate_to_file(gen_test, path_to_test)


if __name__ == '__main__':
    main()
