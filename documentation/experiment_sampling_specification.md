How to create experiments using sampling, instead of all factorial combinations.

1) We start with ground truth examples.
2) Next, we create baseline experiments.
   1) For each ground truth example, we have 2 baseline experiments, one zero-shot and one n-shot experiment.
   2) We only add these experiments to the experiments table if no matching experiment already exists.
3) Next, we create persona-injected experiments.
   1) For each baseline experiment, we get its experiment number.
   2) Set the random seed to the experiment number.
   3) Randomly (based upon the random seed from the previous step) select 10 personas (from the dozens in the persona table) to generate persona-injected experiments.
   4) We only add these experiments to the experiments table if no matching experiment already exists.
4) Next we create bias-mitigation experiments.
   1) For each persona-injected experiment, we get its experiment number.
   2) For each bias mitigation strategy, we generate a bias-mitigation experiment.
   3) We only add these experiments to the experiments table if no matching experiment already exists.
