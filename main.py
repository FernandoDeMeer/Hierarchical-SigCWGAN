print('Start of Hierarchical GAN Training')

from src.training_and_evaluation.train_hierarchical_gan import main

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Meta parameters
    parser.add_argument('-base_dir', default='src/generated_data', type=str)
    parser.add_argument('-use_cuda', type=bool, default=True)
    parser.add_argument('-num_seeds', default=1, type=int)
    parser.add_argument('-initial_seed', default=0, type=int)
    parser.add_argument('-datasets', default=['futures', ], nargs="+")
    parser.add_argument('-algos', default=['HierSigCWGAN', ], nargs="+")

    # Algo hyperparameters
    parser.add_argument('-batch_size', default=100, type=int)
    parser.add_argument('-p', default=3, type=int)
    parser.add_argument('-q', default=3, type=int)
    parser.add_argument('-hidden_dims', default=3 * (50,), type=tuple)
    parser.add_argument('-total_steps', default=5000, type=int)

    args = parser.parse_args()

    main(args)



