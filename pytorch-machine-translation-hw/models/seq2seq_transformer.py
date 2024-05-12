import torch
from torch import nn
import metrics
import math
from torch.optim .lr_scheduler import StepLR

class PositionalEncoding(nn.Module):
    def __init__(self,
                 device,
                 embedding_size,
                 dropout,
                 maxlen=15):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, embedding_size, 2)* math.log(10000) / embedding_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, embedding_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0).to(device)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])
    


class Seq2SeqTransformer(torch.nn.Module):
    def __init__(self, 
                 device,
                 embedding_size,
                 num_encoder_layers,
                 dim_feedforward,
                 src_voc_size,
                 trg_voc_size,
                 target_tokenizer,
                 source_tokenizer,
                 lr_decay_step,
                 lr=1e-4,
                 lr_decay=0.1,
                 dropout_rate=0.1,
                 ):
        super(Seq2SeqTransformer, self).__init__()
        self.device = device
        self.special_tokens = {"pad" : 0, "bos" : 1, "eos" : 2}
       # TODO: Реализуйте конструктор seq2seq трансформера - матрица эмбеддингов, позиционные эмбеддинги, encoder/decoder трансформер, vocab projection head
        
        self.src_emb = nn.Embedding(src_voc_size, embedding_size).to(self.device)
        self.trg_emb = nn.Embedding(trg_voc_size, embedding_size).to(self.device)
        self.positional_encoding = PositionalEncoding(self.device, embedding_size, dropout=dropout_rate)
        
        self.transformer = nn.Transformer(d_model=embedding_size,
                                        num_encoder_layers=num_encoder_layers, 
                                        num_decoder_layers=0, 
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout_rate, batch_first=True).encoder.to(self.device)

        self.vocab_projection_layer = nn.Linear(embedding_size, trg_voc_size).to(self.device) 
        
        self.loss = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.transformer.parameters()}
            ], lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=lr_decay_step, gamma=lr_decay)

        self.target_tokenizer = target_tokenizer
        self.source_tokenizer = source_tokenizer


    def create_masks(self, src):
        src_mask = torch.zeros((src.shape[1], src.shape[1]), device=self.device).type(torch.bool)

        src_pad_mask = (src == self.special_tokens["pad"])
        return src_mask, src_pad_mask
    

    def forward(self, src):
        # TODO: Реализуйте forward pass для модели, при необходимости реализуйте другие функции для обучения
        src_mask, src_pad_mask = self.create_masks(src)
        src_emb = self.positional_encoding(self.src_emb(src))

        out = self.transformer(src_emb, src_mask, src_pad_mask)

        logits = self.vocab_projection_layer(out)
        preds = torch.argmax(logits, dim=2)
                            
        return preds, logits 

 
    def training_step(self, batch):
        # TODO: Реализуйте обучение на 1 батче данных по примеру seq2seq_rnn.py
        self.optimizer.zero_grad()

        input, target = batch
        _, logits = self.forward(input)

        loss = self.loss(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

        loss.backward()
        self.optimizer.step()

        return loss.item() 

    def validation_step(self, batch):
        # TODO: Реализуйте оценку на 1 батче данных по примеру seq2seq_rnn.py
        input, target = batch

        with torch.no_grad():
            _, logits = self.forward(input)

            loss = self.loss(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

        return loss.item()
    
    def predict(self, sentences):

        src_tokenized = torch.tensor([self.source_tokenizer(s) for s in sentences]).to(self.device)
        preds = self.forward(src_tokenized)[0].cpu().detach().numpy()
        translation = [self.target_tokenizer.decode(i) for i in preds]

        return translation


    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.detach().cpu().numpy()
        actuals = target_tensor.detach().cpu().numpy()

        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences




