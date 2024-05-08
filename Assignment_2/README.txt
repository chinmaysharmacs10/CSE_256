The code is structured in the following way:
=> main.py is has the main method
=> Code for the classifier model is in classifier.py
=> Implementation of the encoder, decoder, multi-head attention and feed-forward network is in transformer.py
=> Implementation of encoder, decoder and multi-head attention with ALiBi is in transformer_alibi.py

To run, execute: python main.py --<part_num>
Replace <part_num> with part1, part2 or part3 for whichever part of the assignment you wish to run

Additional flags:
--sanity_check: call utilities.sanity_check post model training
--plot_results: plot training & test accuracy/perplexity of the model

Therefore, if you wish to run part 1 with sanity check and plot the results, execute:
python main.py --part1 --sanity_check --plot_results
