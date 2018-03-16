import unicodedata
import string
import re
import random
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

USE_CUDA = True

SOS_token = 0
EOS_token = 1
PAD_token = 2

train_datafile = "1000.txt"
test_datafile = "half_test_1000.txt"

MAX_LENGTH = 10

test_size = 500

batch_size = 16

convergence_value = 0.0001

# True if in training, False if in evaluating.
to_train = True

# Only relevant if to_train is true.
# True if evaluating a random pair, False if sentence from user.
random_datum = False

# Configuring training
n_epochs = 500000
plot_every = 50
print_every = 100

bins = [7, 12, 52, 102, 1002] # Everything is a normal number + 2 for SOS and EOS.

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3 # Count SOS, EOS, and PAD

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def prepare_string(s):
    s = normalize_string(s)
    l = s.split(' ')
    return l

# This makes sure that the length of each line is exactly the
# value in its bin.

def pad(l):
    l = ["SOS"] + l + ["EOS"]
    for b in bins:
        if len(l) == b:
            return l
        elif len(l) < b:
            return l + ["PAD"] * (b - len(l))

def read_sentences(datafile):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(datafile).read().strip().split('\n')

    # normalize
    data = [pad(prepare_string(l)) for l in lines]

    return data

def prepare_data(vocab, datafile):
    data = read_sentences(datafile)
    print("Read %s sentences" % len(data))

    print("Indexing words...")
    for datum in data:
        vocab.index_words(datum)

    ret_data = [[] for i in range(len(bins))]

    for i in range(len(bins)):
        for datum in data:
            if len(datum) == bins[i]:
                ret_data[i].append(datum)

    num_sorted = 0
    for b in ret_data:
        num_sorted += len(b)
    print("Sorted %d data." % num_sorted)

    return vocab, ret_data

vocab = Lang("Script Vocab")

vocab, data = prepare_data(vocab, train_datafile)
vocab, test_data = prepare_data(vocab, test_datafile)

# Return a list of indexes, one for each word in the sentence
def indexes_from_sentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]

def variable_from_sentence(vocab, sentence):
    indexes = indexes_from_sentence(vocab, sentence)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    #print('var =', var)
    if USE_CUDA: var = var.cuda()
    return var

def variable_from_indexes(seq):
    var = Variable(torch.LongTensor(seq).view(-1, 1))
    if USE_CUDA: var = var.cuda()
    return var

def variable_from_datum(datum):
    input_variable = variable_from_sentence(vocab, datum)
    return input_variable

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden

    def init_cell(self):
        cell = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: cell = cell.cuda()
        return cell

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers=1):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(hidden_size, embedding_size, n_layers)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        # output = F.softmax(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden

    def init_cell(self):
        cell = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: cell = cell.cuda()
        return cell

def test(sentence, total_length, encoder, decoder, max_length = MAX_LENGTH):
    input_variable = variable_from_sentence(vocab, sentence)
    input_length = input_variable.size()[0]

    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_cell = encoder.init_cell()
    encoder_outputs, (encoder_hidden, encoder_cell) = encoder(input_variable, (encoder_hidden, encoder_cell))

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell

    loss = 0 # Added onto for each word

    # TODO: allow for different lengths

    # Run through decoder
    for di in range(total_length):
        decoder_output, (decoder_hidden, decoder_cell) = decoder(decoder_input, (decoder_hidden, decoder_cell))

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        loss += criterion(decoder_output, input_variable[di])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
    return loss/total_length

# Train!

teacher_forcing_ratio = 0.5
clip = 5.0

def train(input_variable, total_length, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_cell = encoder.init_cell()
    encoder_outputs, (encoder_hidden, encoder_cell) = encoder(input_variable, (encoder_hidden, encoder_cell))

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    decoder_cell = encoder_cell
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # TODO: don't require that output havve the same size as the input.

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(total_length):
            decoder_output, (decoder_hidden, decoder_cell) = decoder(decoder_input, (decoder_hidden, decoder_cell))
            loss += criterion(decoder_output, input_variable[di])
            decoder_input = input_variable[di] # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(total_length):
            decoder_output, (decoder_hidden, decoder_cell) = decoder(decoder_input, (decoder_hidden, decoder_cell))
            loss += criterion(decoder_output, input_variable[di])
            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / total_length

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def evaluate(sentence, total_length):
    input_variable = variable_from_sentence(vocab, sentence)
    input_length = input_variable.size()[0]

    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_cell = encoder.init_cell()
    encoder_outputs, (encoder_hidden, encoder_cell) = encoder(input_variable, (encoder_hidden, encoder_cell))

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell

    decoded_words = []

    # TODO: allow other lengths

    # Run through decoder
    for di in range(total_length):
        decoder_output, (decoder_hidden, decoder_cell) = decoder(decoder_input, (decoder_hidden, decoder_cell))

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        decoded_words.append(vocab.index2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
    return decoded_words

def evaluate_randomly(bin_i):
    datum = random.choice(data[bin_i])

    output_words = evaluate(datum, bins[bin_i] + 2) # +2 for SOS and EOS
    output_sentence = ' '.join(output_words)

    print('>', datum)
    print('<', output_sentence)
    print('')

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

if to_train:
    embedding_size = 10
    hidden_size = 10
    n_layers = 2

    # Initialize models
    encoder = EncoderRNN(vocab.n_words, embedding_size, hidden_size, n_layers)
    decoder = DecoderRNN(vocab.n_words, embedding_size, hidden_size, n_layers)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    # Initialize optimizers and criterion
    learning_rate = 0.0001
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every

    # TODO: implement abilities for more bins.

    bin_i = 0

    total_length = bins[bin_i]

    # Begin!
    for epoch in range(1, n_epochs+1):
        if epoch % 500 == 0:
            print("On epoch %d" % epoch)
        # Get training data for this cycle
        '''input_variables = []
        for i in range(batch_size):
            input_variables.append(variable_from_datum(random.choice(data[bin_i])))'''
        input_variable = variable_from_datum(random.choice(data[bin_i]))
        # Run the train function
        loss = train(input_variable, total_length, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss
        if epoch == 1:
            test_loss = []
            for i in range(len(test_data[bin_i])):
                testing_input = test_data[bin_i][i]
                test_loss.append(test(testing_input, total_length, encoder, decoder))
            prev_avg_test_loss = (sum(test_loss)/len(test_loss)).data[0]
            all_avg_test_loss = [prev_avg_test_loss]
        if epoch % print_every == 0:
            test_loss = []
            for i in range(len(test_data[bin_i])):
                testing_input = test_data[bin_i][i]
                test_loss.append(test(testing_input, total_length, encoder, decoder))
            avg_test_loss = (sum(test_loss)/len(test_loss)).data[0]
            print("Average test loss:")
            print(avg_test_loss)
            all_avg_test_loss.append(avg_test_loss)
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)
            if abs(prev_avg_test_loss - avg_test_loss) < convergence_value:
                print("Average test losses:")
                print(all_avg_test_loss)
                break
            prev_avg_test_loss = avg_test_loss

            plot_loss_avg = plot_loss_total / print_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            torch.save(encoder, 'encoder.pt')
            torch.save(decoder, 'decoder.pt')
            evaluate_randomly(bin_i)

    torch.save(encoder, 'encoder.pt')
    torch.save(decoder, 'decoder.pt')

    print(plot_losses)

    show_plot(plot_losses)

    evaluate_randomly(bin_i)
elif random_datum:
    bin_i = 0

    total_length = bins[bin_i] + 2 # +2 for SOS and EOS.

    encoder = torch.load('encoder.pt')
    decoder = torch.load('decoder.pt')
    evaluate_randomly(bin_i)
else:
    bin_i = 0

    total_length = bins[bin_i] + 2 # +2 for SOS and EOS.

    encoder = torch.load('encoder.pt')
    decoder = torch.load('decoder.pt')

    words = input("Please enter a sentence: ")
    output_words = evaluate(words, total_length)
    output_sentence = ' '.join(output_words)

    print(output_sentence + "\n")






















