from data import dataset, generators, robots


def main():
    robot = robots.arm6dof()
    size = 1000
    train_test_split = 0.8

    gen_train = generators.RandomGen(robot=robot, batch_size=32, n=int(size * train_test_split))
    gen_test = generators.RandomGen(robot=robot, batch_size=32, n=int(size * (1 - train_test_split)))

    dataset.generate_to_file(gen_train, '../data/train_fk_ds_0_0_2.csv')
    dataset.generate_to_file(gen_test, '../data/test_fk_ds_0_0_2.csv')


if __name__ == '__main__':
    main()
