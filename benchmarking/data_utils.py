import pandas as pd
import tempfile
import os

def net_to_dne_string(net):
    """Export a Netica Net to its DNE-format string by writing to temp file."""
    with tempfile.NamedTemporaryFile(suffix=".dne", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        net.write(tmp_path)
        with open(tmp_path, "r") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def dne_string_to_net(dne_str):
    """Reconstruct a Netica Net from a DNE-format string."""
    from bni_netica.bni_netica import Net
    with tempfile.NamedTemporaryFile(suffix=".dne", delete=False, mode="w") as tmp:
        tmp.write(dne_str)
        tmp_path = tmp.name
    try:
        net = Net(tmp_path)  # assuming .read(path) works
        net.compile()
        return net
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def save_nets_to_parquet(nets, path="nets.parquet"):
    records = [{"name": net.name(), "dne": net_to_dne_string(net)} for net in nets]
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    print(f"Saved {len(nets)} nets to {path}")

def load_nets_from_parquet(path="nets.parquet"):
    df = pd.read_parquet(path)
    nets = []
    for _, row in df.iterrows():
        net = dne_string_to_net(row["dne"])
        nets.append(net)
    print(f"Loaded {len(nets)} nets from {path}")
    return nets