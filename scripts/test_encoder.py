from src.models.rnafm_encoder import RNAFMEncoder

encoder = RNAFMEncoder(device="cpu")

seq = ["ACGUACGUACGUACGU"]

emb = encoder(seq)

print(emb.shape)