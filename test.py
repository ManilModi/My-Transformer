from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import torch
from torch import nn
from Transformer import Transformer
from TransformerLRScheduler import TransformerLRScheduler
from LabelSmoothingLoss import LabelSmoothingLoss
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("opus_books", "de-en", split="train")
print(dataset[0])  # {'translation': {'de': ..., 'en': ...}}



SRC_LANG = "de"
TGT_LANG = "en"

tokenizer_src = get_tokenizer('spacy', language='de_core_news_sm')
tokenizer_tgt = get_tokenizer('spacy', language='en_core_web_sm')

def yield_tokens(data_iter, lang):
    tokenizer = tokenizer_src if lang == SRC_LANG else tokenizer_tgt
    for sample in data_iter:
        yield tokenizer(sample["translation"][lang])


# Build vocabularies
vocab_src = build_vocab_from_iterator(
    yield_tokens(dataset, SRC_LANG),
    specials=['<unk>', '<pad>', '<bos>', '<eos>']
)
vocab_src.set_default_index(vocab_src['<unk>'])

vocab_tgt = build_vocab_from_iterator(
    yield_tokens(dataset, TGT_LANG),
    specials=['<unk>', '<pad>', '<bos>', '<eos>']
)
vocab_tgt.set_default_index(vocab_tgt['<unk>'])

PAD_IDX = vocab_src['<pad>']  # Consistent with source padding index

VOCAB_SIZE_SRC = len(vocab_src)
VOCAB_SIZE_TGT = len(vocab_tgt)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    src_vocab_size=VOCAB_SIZE_SRC,
    tgt_vocab_size=VOCAB_SIZE_TGT,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    dropout=0.1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = TransformerLRScheduler(optimizer, d_model=512)
loss_fn = LabelSmoothingLoss(
    label_smoothing=0.1,
    vocab_size=VOCAB_SIZE_TGT,
    ignore_index=PAD_IDX
)

PAD_IDX = vocab_tgt['<pad>']
BOS_IDX = vocab_tgt['<bos>']
EOS_IDX = vocab_tgt['<eos>']



def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for example in batch:
        src_tokens = [vocab_src['<bos>']] + vocab_src(tokenizer_src(example['translation'][SRC_LANG])) + [vocab_src['<eos>']]
        tgt_tokens = [vocab_tgt['<bos>']] + vocab_tgt(tokenizer_tgt(example['translation'][TGT_LANG])) + [vocab_tgt['<eos>']]

        src_batch.append(torch.tensor(src_tokens, dtype=torch.long))
        tgt_batch.append(torch.tensor(tgt_tokens, dtype=torch.long))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)

    return {'src': src_batch, 'tgt': tgt_batch}

from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz), dtype=torch.bool)).transpose(0, 1)
    return mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))


def train_step(model, batch, optimizer, scheduler, loss_fn, device):
    model.train()
    src = batch['src'].to(device)
    tgt = batch['tgt'].to(device)

    # Prepare decoder input and target
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

    logits = model(src, tgt_input, tgt_mask=tgt_mask)
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    return loss.item()


def train_model(model, dataloader, optimizer, scheduler, loss_fn, device, num_epochs=5):
    model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            loss = train_step(model, batch, optimizer, scheduler, loss_fn, device)
            total_loss += loss

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss:.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f">>> Epoch [{epoch+1}] Avg Loss: {avg_loss:.4f}")


batch = next(iter(train_dataloader))
src = batch['src'].to(device)
tgt = batch['tgt'].to(device)
tgt_inp = tgt[:, :-1]
tgt_out = tgt[:, 1:]

logits = model(src, tgt_inp)
logits_flat = logits.view(-1, logits.size(-1))
tgt_out_flat = tgt_out.reshape(-1)

print("Max target idx:", tgt_out_flat.max().item(), "/", VOCAB_SIZE_TGT)

loss = loss_fn(logits_flat, tgt_out_flat)
print("Initial Loss:", loss.item())

loss.backward()


model = Transformer(
    src_vocab_size=VOCAB_SIZE_SRC,
    tgt_vocab_size=VOCAB_SIZE_TGT,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    dropout=0.1
).to(device)


if __name__ == "__main__":
    
    train_model(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        num_epochs=5
    )
