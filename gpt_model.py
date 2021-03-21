import json
import os
import tensorflow as tf
import torch

from pathlib import Path
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME


class GPTModel:
    '''
    Class representing the GPT model implementation, featuring:
    - Training the model
    - Saving all intermediate data for later reuse
    - Saving the trained model
    - Parameterized prediction using the trained or loaded model
    '''

    DEFAULT_MODEL_PATH = './model/'
    TOKENIZER_MODEL_DIR = DEFAULT_MODEL_PATH + 'tokenized_data/'
    CUMULATIVE_STRING_FILE = DEFAULT_MODEL_PATH + 'cumulative_string.json'
    GPT_MODEL_DIR = DEFAULT_MODEL_PATH + 'gpt_data/'

    def __init__(self):
        '''
        Initialize empty model
        '''

        self.tokenizer = None
        self.model = None

    @staticmethod
    def load_or_train_tokenizer(file_paths, tokenizer_mode_path):
        '''
        Tries to load saved text tokenizer
        If there is none, trains the new tokenizer and saves is
        '''

        if not os.path.exists(tokenizer_mode_path):
            print('Tokenizer model not found, training one')

            from tokenizers.models import BPE
            from tokenizers import Tokenizer
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
            from tokenizers.normalizers import NFKC, Sequence
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.trainers import BpeTrainer

            tokenizer = Tokenizer(BPE())
            tokenizer.normalizer = Sequence([
                NFKC()
            ])
            tokenizer.pre_tokenizer = ByteLevel()
            tokenizer.decoder = ByteLevelDecoder()

            trainer = BpeTrainer(
                vocab_size=50000,
                show_progress=True,
                inital_alphabet=ByteLevel.alphabet(),
                special_tokens=[
                    "<s>",
                    "<pad>",
                    "</s>",
                    "<unk>",
                    "<mask>"
                ]
            )
            tokenizer.train(file_paths, trainer)

            if not os.path.exists(tokenizer_mode_path):
                os.makedirs(tokenizer_mode_path)
            tokenizer.model.save(tokenizer_mode_path, None)

        print('Loading trained tokenizer model')

        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_mode_path)
        tokenizer.add_special_tokens({
            'eos_token': '</s>',
            'bos_token': '<s>',
            'unk_token': '<unk>',
            'pad_token': '<pad>',
            'mask_token': '<mask>'
        })

        return tokenizer

    @staticmethod
    def load_or_tokenize_cumulative_string(tokenizer, file_paths, cumulative_string_path):
        '''
        Tries to load saved tokenized string, with all provided data concatenated
        If there is none, generates new cumulative string and saves it
        '''

        if os.path.exists(cumulative_string_path):
            print('Loading existing cumulative string')

            with open(cumulative_string_path, 'r') as in_file:
                return json.load(in_file)

        else:
            print('Creating and tokenizing the cumulative string')

            single_string = ''
            for filename in file_paths:
                with open(filename, 'r', encoding='utf-8') as in_file:
                    x = in_file.read()
                    single_string += x + tokenizer.eos_token
            string_tokenized = tokenizer.encode(single_string)

            with open(cumulative_string_path, 'w+') as out_file:
                json.dump(string_tokenized, out_file)

            return string_tokenized

    @staticmethod
    def load_or_generate_dataset(tokenizer, file_paths, cumulative_string_path):
        '''
        Acquires the tokenized cumulative sting through loading or generating it
        And than creates the randomized dataset using it
        '''

        # Create a single string from all documents and tokenize it
        cumulative_string_tokenized = GPTModel.load_or_tokenize_cumulative_string(tokenizer, file_paths, cumulative_string_path)

        print('Generating the dataset')

        # Generate actural training set
        examples = []
        block_size = 100
        BATCH_SIZE = 12
        BUFFER_SIZE = 1000

        for i in range(0, len(cumulative_string_tokenized) - block_size + 1, block_size):
            examples.append(cumulative_string_tokenized[i:i + block_size])
        inputs, labels = [], []

        for ex in examples:
            inputs.append(ex[:-1])
            labels.append(ex[1:])

        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        return dataset

    @staticmethod
    def load_or_train_model(tokenizer, file_paths, gpt_model_path, cumulative_string_path):
        '''
        Tries to load previously trained model
        If there is none, runs the training on the generated dataset
        '''

        if os.path.exists(gpt_model_path):
            print('Loading GPT model')

            return TFGPT2LMHeadModel.from_pretrained(gpt_model_path + 'pytorch_model.bin')

        else:

            print('GPT model not found, training one')

            # Creating the configurations from which the model can be made
            config = GPT2Config(
                vocab_size=tokenizer.vocab_size,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # Creating the model
            model = TFGPT2LMHeadModel(config)

            # Defining our optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)

            # Definining our loss function
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            # Defining our metric which we want to observe
            metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

            # Compiling the model
            model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])

            # Prepare training dataset
            dataset = GPTModel.load_or_generate_dataset(tokenizer, file_paths, cumulative_string_path)

            # Execute training
            num_epoch = 3
            model.fit(dataset, epochs=num_epoch)

            # Creating directory if it is not present
            os.mkdir(gpt_model_path)

            # Save the model
            output_model_file = os.path.join(gpt_model_path, WEIGHTS_NAME)
            model.save_pretrained(output_model_file)

            # Save the config
            model_to_save = model.module if hasattr(model, 'module') else model
            output_config_file = os.path.join(gpt_model_path, CONFIG_NAME)
            model_to_save.config.to_json_file(output_config_file)

            return model

    def load_or_train(self, file_paths=None, model_path=DEFAULT_MODEL_PATH):
        '''
        Tries to load saved tokenizer and model, or trains them if none are found
        '''

        # Get paths to all base documents
        if file_paths is None:
            file_paths = [str(x) for x in Path('./sessions/sessions_text/').glob('*.txt')]
        print('Num base files found: {}'.format(len(file_paths)))

        if not os.path.exists(model_path):
            os.mkdir(model_path)

        self.tokenizer = self.load_or_train_tokenizer(file_paths, GPTModel.TOKENIZER_MODEL_DIR)

        self.model = self.load_or_train_model(
            self.tokenizer,
            file_paths,
            GPTModel.GPT_MODEL_DIR,
            GPTModel.CUMULATIVE_STRING_FILE)

    def generate(
            self,
            base_text,
            max_text_len=200,
            min_text_len=100,
            top_k=50,
            top_p=0.9):
        '''
        Generates a text sequence with provided staring text and parameters
        '''

        if self.tokenizer is None or self.model is None:
            return ''

        # Encoding the input text
        input_ids = self.tokenizer.encode(base_text, return_tensors='tf')

        # Getting out output
        beam_output = self.model.generate(
            input_ids,
            max_length=max_text_len,
            min_length=min_text_len,
            num_beams=3,
            temperature = 0.7,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=2,
            do_sample=True)

        return self.tokenizer.decode(beam_output[0])


def main():
    '''
    Basic testing utility for the GPTModel
    '''

    import sys

    if len(sys.argv) <= 2:
        print('Usage: python3 {} [input text ...]'.format(sys.argv[0]))
        return

    model = GPTModel()
    model.load_or_train()

    text = ' '.join(sys.argv[1:])
    generated_text = model.generate(text)

    print(generated_text)


if __name__ == '__main__':
    main()
