from data import dataset, generators, robots


def main():
    robot = robots.arm6dof()
    gen_train = generators.RandomGen(robot=robot, batch_size=32, n=8_000)
    gen_test = generators.RandomGen(robot=robot, batch_size=32, n=2_000)

    dataset.generate_to_file(gen_train, '../data/train_fk_ds_0_0_1.csv')
    dataset.generate_to_file(gen_test, '../data/test_fk_ds_0_0_1.csv')


if __name__ == '__main__':
    main()
