import time

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import os
import argparse
import matplotlib.pyplot as plt

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from classifier import Classifier
from utilities import Utilities
from transformer import Decoder
from transformer_alibi import ALiBiDecoder


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels


def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, attention_weights = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    total_loss = 0.0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss, _ = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


def train_classifier(tokenizer, vocab_size, train_CLS_loader, test_CLS_loader, drop_prob=0.0, batch_norm=False,
                     sanity_check=False, part3=False):
    model = Classifier(n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size, vocab_size=vocab_size,
                       n_input=n_input, n_hidden=n_hidden, n_output=n_output, drop_prob=drop_prob,
                       batch_norm=batch_norm, part3=part3)
    m = model.to(device)
    print('\nNumber of Parameters in the classifier: ', sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')
    # print('\n', m)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    size = len(train_CLS_loader.dataset)
    num_batches = len(train_CLS_loader)
    loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    start_time = time.time()
    for epoch in range(epochs_CLS):
        train_loss, correct = 0, 0

        for xb, yb in train_CLS_loader:
            optimizer.zero_grad()
            xb, yb = xb.to(device), yb.to(device)
            predictions, _ = model(xb)
            loss = loss_fn(predictions, yb)
            train_loss += loss.item()
            correct += (predictions.argmax(1) == yb).type(torch.float).sum().item()
            loss.backward()
            optimizer.step()

        # scheduler.step()

        average_train_loss = train_loss / num_batches
        loss_list.append(average_train_loss)
        accuracy = correct / size
        train_accuracy_list.append(accuracy)
        test_accuracy = compute_classifier_accuracy(model, test_CLS_loader)
        test_accuracy_list.append(test_accuracy)
        print("\nEpoch #: {}".format(epoch + 1))
        print("Training loss: {}".format(average_train_loss))
        print("Training accuracy: {}".format(accuracy))
        print("Classifier Accuracy on test set: ", test_accuracy)

    end_time = time.time()
    print('\nTime taken to train classifier: {}'.format(end_time - start_time))

    if sanity_check:
        utility = Utilities(tokenizer=tokenizer, model=model)
        utility.sanity_check(sentence="These virtues give me an unshakable faith in America", block_size=block_size)

    return loss_list, train_accuracy_list, test_accuracy_list


def train_decoder(tokenizer, vocab_size, train_LM_loader, text_files_path, ff_dim, part3=False):

    def get_test_perplexity(model, tokenizer, text_files_path, file_name):
        input_file = text_files_path + "/test_LM_" + str(file_name).lower() + ".txt"
        with open(input_file, 'r', encoding='utf-8') as f:
            lm_test_text = f.read()
        test_lm_dataset = LanguageModelingDataset(tokenizer, lm_test_text, block_size)
        test_lm_loader = DataLoader(test_lm_dataset, batch_size=batch_size, shuffle=True)
        test_perplexity = compute_perplexity(model, test_lm_loader, eval_iters)
        print("Perplexity for {}: {:.4f}".format(file_name, test_perplexity))
        return test_perplexity

    model = Decoder(n_embd=n_embd, num_heads=n_head, num_layers=n_layer, block_size=block_size, vocab_size=vocab_size,
                    ff_dim=ff_dim)
    if part3:
        model = ALiBiDecoder(n_embd=n_embd, num_heads=n_head, num_layers=n_layer, block_size=block_size,
                             vocab_size=vocab_size, expansion_factor=1, causal=True)
    m = model.to(device)
    print('\nNumber of Parameters in the decoder: ', sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')
    # print('\n', m)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    train_perplexity_list = []
    obama_perplexity_list = []
    hbush_perplexity_list = []
    wbush_perplexity_list = []

    start_time = time.time()

    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break

        xb, yb = xb.to(device), yb.to(device)
        loss, _ = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (i + 1) % eval_interval == 0:
            train_perplexity = compute_perplexity(model, train_LM_loader, eval_iters)
            train_perplexity_list.append(train_perplexity)
            print("\n=> Iter {}".format(i+1))
            print("Train perplexity: {:.4f}".format(train_perplexity))
            obama_perplexity_list.append(get_test_perplexity(model, tokenizer, text_files_path, "obama"))
            hbush_perplexity_list.append(get_test_perplexity(model, tokenizer, text_files_path, "hbush"))
            wbush_perplexity_list.append(get_test_perplexity(model, tokenizer, text_files_path, "wbush"))

    end_time = time.time()
    print('\nTime taken to train decoder: {}'.format(end_time - start_time))
    return train_perplexity_list, obama_perplexity_list, hbush_perplexity_list, wbush_perplexity_list


def parse_input():
    parser = argparse.ArgumentParser(prog="PA2", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_defaults(func=lambda _: parser.print_help())

    parser.add_argument(
        '--part1',
        action='store_true',
        help='Flag for Part 1: Encoder Trained With Classifier'
    )
    parser.add_argument(
        '--part2',
        action='store_true',
        help='Flag for Part 2: Pretraining Decoder Language Model'
    )
    parser.add_argument(
        '--part3',
        action='store_true',
        help='Flag for Part 3: Exploration'
    )
    parser.add_argument(
        '--plot_results',
        action='store_true',
        help='Flag to plot graphs'
    )
    parser.add_argument(
        '--sanity_check',
        action='store_true',
        help='Flag to enable sanity check'
    )
    return parser.parse_args()


def main():
    args = parse_input()

    print("Loading data and creating tokenizer ...")
    text_files_path = '/Users/chinmaysharma/Documents/UCSD_Courses/Spring_2024/CSE_256/Programming_Assignments/Assignment_2/speechesdataset'
    texts = load_texts(text_files_path)
    tokenizer = SimpleTokenizer(' '.join(texts))    # create a tokenizer from the data
    vocab_size = tokenizer.vocab_size
    print("Vocabulary size is", vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, text_files_path + "/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, text_files_path + "/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

    inputfile = text_files_path + "/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    epochs = [i for i in range(1, 16)]
    eval_intervals = [i for i in range(100, 501, 100)]

    if args.part1:
        print("\nRunning Part 1: Encoder Trained With Classifier...")
        loss_list, train_accuracy_list, test_accuracy_list = train_classifier(
            tokenizer, vocab_size, train_CLS_loader, test_CLS_loader, sanity_check=args.sanity_check)

        if args.plot_results:
            plt.plot(epochs, train_accuracy_list)
            plt.xlabel('Epochs')
            plt.ylabel('Training Accuracy')
            plt.title('Plot of Training Accuracy vs Epochs')
            plt.savefig(f"part_1_training_accuracy.png")
            plt.clf()

            plt.plot(epochs, test_accuracy_list)
            plt.xlabel('Epochs')
            plt.ylabel('Test Accuracy')
            plt.title('Plot of Test Accuracy vs Epochs')
            plt.savefig(f"part_1_test_accuracy.png")
            plt.clf()

            plt.plot(epochs, loss_list)
            plt.xlabel('Epochs')
            plt.ylabel('Training Loss')
            plt.title('Plot of Training Loss vs Epochs')
            plt.savefig(f"part_1_training_loss.png")

    if args.part2:
        print("\nRunning Part 2: Pretraining Decoder Language Model...")
        train_perplexity_list, obama_perplexity_list, hbush_perplexity_list, wbush_perplexity_list = train_decoder(
            tokenizer, vocab_size, train_LM_loader, text_files_path, ff_dim=(4 * n_embd))

        if args.plot_results:
            plt.plot(eval_intervals, train_perplexity_list)
            plt.xlabel('Eval Interval')
            plt.ylabel('Training Perplexity')
            plt.title('Plot of Training Perplexity vs Eval Intervals')
            plt.savefig(f"part_2_training_perplexity.png")
            plt.clf()

            plt.plot(eval_intervals, obama_perplexity_list, label='Obama')
            plt.plot(eval_intervals, hbush_perplexity_list, label='H. Bush')
            plt.plot(eval_intervals, wbush_perplexity_list, label='W. Bush')
            plt.xlabel('Eval Interval')
            plt.ylabel('Test Perplexity')
            plt.title('Plot of Obama, H. Bush, W. Bush Test Perplexity vs Eval Intervals')
            plt.legend()
            plt.savefig(f"part_2_test_perplexity.png")

    if args.part3:
        print("\nRunning Part 3: Architectural Exploration...")
        loss_list, train_accuracy_list, test_accuracy_list = train_classifier(
           tokenizer, vocab_size, train_CLS_loader, test_CLS_loader, part3=True)

        train_perplexity_list, obama_perplexity_list, hbush_perplexity_list, wbush_perplexity_list = train_decoder(
            tokenizer, vocab_size, train_LM_loader, text_files_path, ff_dim=(4 * n_embd), part3=True)

        if args.plot_results:
            plt.plot(epochs, train_accuracy_list)
            plt.xlabel('Epochs')
            plt.ylabel('Training Accuracy')
            plt.title('Plot of Training Accuracy vs Epochs')
            plt.savefig(f"part_3_training_accuracy.png")
            plt.clf()

            plt.plot(epochs, test_accuracy_list)
            plt.xlabel('Epochs')
            plt.ylabel('Test Accuracy')
            plt.title('Plot of Test Accuracy vs Epochs')
            plt.savefig(f"part_3_test_accuracy.png")
            plt.clf()

            plt.plot(epochs, loss_list)
            plt.xlabel('Epochs')
            plt.ylabel('Training Loss')
            plt.title('Plot of Training Loss vs Epochs')
            plt.savefig(f"part_3_training_loss.png")
            plt.clf()

            plt.plot(eval_intervals, train_perplexity_list)
            plt.xlabel('Eval Interval')
            plt.ylabel('Training Perplexity')
            plt.title('Plot of Training Perplexity vs Eval Intervals')
            plt.savefig(f"part_3_training_perplexity.png")
            plt.clf()

            plt.plot(eval_intervals, obama_perplexity_list, label='Obama')
            plt.plot(eval_intervals, hbush_perplexity_list, label='H. Bush')
            plt.plot(eval_intervals, wbush_perplexity_list, label='W. Bush')
            plt.xlabel('Eval Interval')
            plt.ylabel('Test Perplexity')
            plt.title('Plot of Obama, H. Bush, W. Bush Test Perplexity vs Eval Intervals')
            plt.legend()
            plt.savefig(f"part_3_test_perplexity.png")

        print("\nRunning Part 3: Performance Improvement...")


if __name__ == "__main__":
    main()
