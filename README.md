# Quickstart

### Secrets

After cloning the repo if you want to run evals (especially the toxicity eval), make sure to first create the `.env` file (`cp .env-template .env`) and fill the correct fields

### Running training

Training is very modular, configs handled by [hydra](https://hydra.cc/docs/configure_hydra/intro/). Check the default config at `config/train_config/defaults`

To run training:

`python3 main.py <any overrides>`

For example, by default `sft` is ran with the `llama_3_1b_instruct` model. To run both `sft` and `dpo` with `gemma_2b`:

`python3 main.py training=all training.model=gemma_2b`

### Running eval

Same as training, configs are handled by [hydra](https://hydra.cc/docs/configure_hydra/intro/). Unfortunately, choosing which evals to run manually is a bit more annoying than whatever we had to do in the training setup. 

`python3 eval.py 'defaults=[_self_, logger: wandb_disabled, evals/some-eval1@evals.eval1, evals/some-eval2@evals.eval2, ...]'`

Where `...` is optionally whatever additional eval you want to run. To be honest, it's much easier to just set which evals you want to run in the `config/eval_config/default.yaml` file and just run `python3 eval.py`.
