# Makemore
Makemore takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an autoregressive character-level language model, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT). For example, feed it a database of names, and makemore will generate cool name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company.

Here are some unique baby names that get eventually generated from current default settings (test logprob of ~2.02, though much lower logprobs are achievable with some hyperparameter tuning):
novani
juliah
khalia
mareel
graith
delaniah
wellasia
arra
kiahli
shelen
