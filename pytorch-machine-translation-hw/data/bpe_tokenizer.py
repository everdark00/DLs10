from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


class BPETokenizer:
    def __init__(self, sentence_list):
        """
        sentence_list - список предложений для обучения
        """
        # TODO: Реализуйте конструктор c помощью https://huggingface.co/docs/transformers/fast_tokenizers, обучите токенизатор, подготовьте нужные аттрибуты(word2index, index2word)
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.enable_truncation(max_length=15)
        self.tokenizer.enable_padding(length=15)
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
        )

        self.tokenizer.train_from_iterator(iter(sentence_list), trainer, len(sentence_list))

    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        # TODO: Реализуйте метод токенизации с помощью обученного токенизатора
        return self.tokenizer.encode(sentence).ids


    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        # TODO: Реализуйте метод декодирования предсказанных токенов

        return self.tokenizer.decode(token_list).split()