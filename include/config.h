struct Config
{
    int block_size = 16;
    int batch_size = 4;
    int d_model = 32;
    int num_heads = 1;
    int num_layers = 1;
    int d_ff = 128;

    int max_steps;
    int eval_interval;

    double learning_rate = 0.05;
    double weight_init_scale = 0.01;

    unsigned int random_seed = 42;
};
