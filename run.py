import torch
from reformer_pytorch import Reformer, LSHAttention, ReformerLM, ReformerEncDec


DE_SEQ_LEN = 4096
EN_SEQ_LEN = 4096

enc_dec = ReformerEncDec(
    dim = 512,
    enc_num_tokens = 20000,
    enc_depth = 6,
    enc_max_seq_len = DE_SEQ_LEN,
    dec_num_tokens = 20000,
    dec_depth = 6,
    dec_max_seq_len = EN_SEQ_LEN
).cuda()

train_seq_in = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long().cuda()
train_seq_out = torch.randint(0, 20000, (1, EN_SEQ_LEN)).long().cuda()
input_mask = torch.ones(1, DE_SEQ_LEN).bool().cuda()

loss = enc_dec(train_seq_in, train_seq_out, return_loss = True, enc_input_mask = input_mask)
loss.backward()
# learn

# evaluate with the following
eval_seq_in = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long().cuda()
eval_seq_out_start = torch.tensor([[0.]]).long().cuda() # assume 0 is id of start token
samples = enc_dec.generate(eval_seq_in, eval_seq_out_start, seq_len = EN_SEQ_LEN, eos_token = 1) # assume 1 is id of stop token
print(samples.shape) # (1, <= 1024) decode the tokens



DE_SEQ_LEN = 4096
EN_SEQ_LEN = 4096

encoder = ReformerLM(
    num_tokens = 20000,
    emb_dim = 128,
    dim = 1024,
    depth = 12,
    heads = 8,
    max_seq_len = DE_SEQ_LEN,
    fixed_position_emb = True,
    return_embeddings = True # return output of last attention layer
).cuda()

decoder = ReformerLM(
    num_tokens = 20000,
    emb_dim = 128,
    dim = 1024,
    depth = 12,
    heads = 8,
    max_seq_len = EN_SEQ_LEN,
    fixed_position_emb = True,
    causal = True
).cuda()

x  = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long().cuda()
yi = torch.randint(0, 20000, (1, EN_SEQ_LEN)).long().cuda()

enc_keys = encoder(x)               # (1, 4096, 1024)
yo = decoder(yi, keys = enc_keys)   # (1, 4096, 20000)


model = Reformer(
    dim = 512,
    depth = 12,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True
).cuda()

x = torch.randn(1, 8192, 512).cuda()
y = model(x) # (1, 8192, 512)


attn = LSHAttention(
    bucket_size = 64,
    n_hashes = 16,
    causal = True
)

qk = torch.randn(10, 1024, 128)
v = torch.randn(10, 1024, 128)

out, attn, buckets = attn(qk, v)

CONTEXT_LEN = 512
SEQ_LEN = 8192

model = ReformerLM(
    num_tokens= 20000,
    dim = 1024,
    depth = 1,
    max_seq_len = SEQ_LEN,
    ff_chunks = 8,
    causal = True
)

c = torch.randn(1, CONTEXT_LEN, 1024)
x = torch.randint(0, 20000, (1, SEQ_LEN)).long()

i_mask = torch.ones(1, SEQ_LEN).bool()
c_mask = torch.ones(1, CONTEXT_LEN).bool()

y = model(x, keys = c, input_mask = i_mask, context_mask = c_mask)