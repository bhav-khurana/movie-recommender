# Movie Recommender System

This is a movie recommender system made using Genetic Algorithm. This is trained on the popular [`MovieLens 100k Dataset`](https://grouplens.org/datasets/movielens/100k/).

This implementation only makes use of the inbuilt features of the `Python` language and the libraries `numpy` and `matplotlib` to data the dataframes and plot graphs respectively. The actual implementation of the GA is done in code, without any supoort of any external library.

## How does the model work

This model tries to calculate the most probable ratings of a user for the given movies. It does so using Genetic Algorithm with the following techniques:

- Integer Number Encoding of chromosomes
- Pearson Relation Coeeficient based fitness function for the individuals
- Tournament Selection for parents
- Uniform Crossover with crossover selection mask
- Random Mutation on real number GA encoding
- Termination based on number of generations, change in leader, and improvement of leader's fitness over generations
