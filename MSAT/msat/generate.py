import argparse
import logging
import pathlib
import pprint
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import tqdm
sys.path.append(r'/work100/weixp/MSMM-main/baseline/MTMT-bar')
import dataset

import representation
import utils

import torch.distributed as dist
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.utils.data.distributed import DistributedSampler
@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-as",
        "--atten_strategy",
        choices=("local", "global"),
        required=False,
        default="global",
        help="dataset key",
    )
    
    parser.add_argument(
        "-d",
        "--dataset",
        default='sod-bar',
        # choices=("sod", "lmd", "lmd_full", "snd"),
        required=False,
        help="dataset key",
    )
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-ns",
        "--n_samples",
        # default=50,
        type=int,
        help="number of samples to generate",
    )
    # Model
    parser.add_argument(
        "-s",
        "--shuffle",
        action="store_true",
        help="whether to shuffle the test data",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    parser.add_argument(
        "--model_steps",
        type=int,
        help="step of the trained model to load (default to the best model)",
    )
    # Sampling
    parser.add_argument(
        "--seq_len", default=1024, type=int, help="sequence length to generate"
    )
    parser.add_argument(
        "--temperature",
        nargs="+",
        default=1.0,
        type=float,
        help="sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default="top_k",
        type=str,
        help="sampling filter (default: 'top_k')",
    )
    parser.add_argument(
        "--filter_threshold",
        nargs="+",
        default=0.9,
        type=float,
        help="sampling filter threshold (default: 0.9)",
    )
    # Others
    parser.add_argument("-g", "--gpu", type=int, help="gpu number")
    parser.add_argument(
        "-j", "--jobs", default=1, type=int, help="number of jobs"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )

    return parser.parse_args(args=args, namespace=namespace)


def save_pianoroll(filename, music, size=None, **kwargs):
    """Save the piano roll to file."""
    music.show_pianoroll(track_label="program", **kwargs)
    if size is not None:
        plt.gcf().set_size_inches(size)
    plt.savefig(filename)
    plt.close()


def save_result(filename, data, sample_dir, encoding,numberm):
    """Save the results in multiple formats."""
    if numberm==1:
        representation.save_csv_codes(sample_dir / "csv" / "truth" / f"{filename}.csv", data)
        music = representation.decode(data, encoding)
        # Save as a MIDI file
        music.write(sample_dir / "mid"  / "truth" / f"{filename}.mid")
    if numberm==2:
        representation.save_csv_codes(sample_dir / "csv" / "4-beats" / f"{filename}.csv", data)
        music = representation.decode(data, encoding)
        # Save as a MIDI file
        music.write(sample_dir / "mid"  / "4-beats" / f"{filename}.mid")
    if numberm==3:
        representation.save_csv_codes(sample_dir / "csv" / "16-beats" / f"{filename}.csv", data)
        music = representation.decode(data, encoding)
        # Save as a MIDI file
        music.write(sample_dir / "mid"  / "16-beats" / f"{filename}.mid")
    if numberm==4:
        representation.save_csv_codes(sample_dir / "csv" / "unconditioned" / f"{filename}.csv", data)
        music = representation.decode(data, encoding)
        # Save as a MIDI file
        music.write(sample_dir / "mid"  / "unconditioned" / f"{filename}.mid")
    if numberm==5:
        representation.save_csv_codes(sample_dir / "csv" / "instrument-beats" / f"{filename}.csv", data)
        music = representation.decode(data, encoding)
        # Save as a MIDI file
        music.write(sample_dir / "mid"  / "instrument-beats" / f"{filename}.mid")

        
def get_cut_beat(beatnumber,batch):
    # cut hang：
    batch1 = []                
    for i in range(batch["seq"].shape[1]):
        if batch["seq"][0][i][0]!=1 and batch["seq"][0][i][0]!=3 and batch["seq"][0][i][0]!=4:
            batch1.append(batch["seq"][0][i])
        elif batch["seq"][0][i][0]==1:
            batch1.append(batch["seq"][0][i])
        elif batch["seq"][0][i][0]==3 and batch["seq"][0][i][1]<=beatnumber:
            batch1.append(batch["seq"][0][i])
    
    for r in batch1:
        r = torch.unsqueeze(r, 0)
        if r[0][0] ==0:
            batch_input = r
            continue
        else:
            batch_input = torch.cat((batch_input,r),0)
    
    batch_input = torch.unsqueeze(batch_input, 0)
    return batch_input
def get_instrument_beat(batch):
    # cut hang：
    batch1 = []                
    for i in range(batch["seq"].shape[1]):
        if batch["seq"][0][i][0]!=3 and batch["seq"][0][i][0]!=4:
            batch1.append(batch["seq"][0][i])

    
    for r in batch1:
        r = torch.unsqueeze(r, 0)
        if r[0][0] ==0:
            batch_input = r
            continue
        else:
            batch_input = torch.cat((batch_input,r),0)
    
    batch_input = torch.unsqueeze(batch_input, 0)
    return batch_input      
      
    
def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.names is None:
            args.names = pathlib.Path(
                f"data/{args.dataset}/processed/test-names.txt"
            )
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes/")
        if args.out_dir is None:
            args.out_dir = pathlib.Path(f"exp/test_{args.dataset}")

    # Set up the logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "generate.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    logging.info(f"Saved arguments to {args.out_dir / 'generate-args.json'}")
    utils.save_args(args.out_dir / "generate-args.json", args)

    # Load training configurations
    logging.info(
        f"Loading training arguments from: {args.out_dir / 'train-args.json'}"
    )
    train_args = utils.load_json(args.out_dir / "train-args.json")
    logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")

    # Make sure the sample directory exists
    sample_dir = args.out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    (sample_dir / "csv").mkdir(exist_ok=True)
    (sample_dir / "csv" / "truth").mkdir(exist_ok=True)
    (sample_dir / "csv" / "4-beats").mkdir(exist_ok=True)
    (sample_dir / "csv" / "16-beats").mkdir(exist_ok=True)
    (sample_dir / "csv" / "instrument-beats").mkdir(exist_ok=True)
    (sample_dir / "csv" / "unconditioned").mkdir(exist_ok=True)
    (sample_dir / "mid").mkdir(exist_ok=True)
    (sample_dir / "mid" / "truth").mkdir(exist_ok=True)
    (sample_dir / "mid" / "4-beats").mkdir(exist_ok=True)
    (sample_dir / "mid" / "instrument-beats").mkdir(exist_ok=True)
    (sample_dir / "mid" / "16-beats").mkdir(exist_ok=True)
    (sample_dir / "mid" / "unconditioned").mkdir(exist_ok=True)



    # Get the specified device
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Load the encoding
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

    # Create the dataset and data loader
    logging.info(f"Creating the data loader...")
    test_dataset = dataset.MusicDataset(
        args.names,
        args.in_dir,
        encoding,
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        use_csv=args.use_csv,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=args.shuffle,
        num_workers=args.jobs,
        collate_fn=dataset.MusicDataset.collate,
    )

    # Create the model
    logging.info(f"Creating the model...")
    model = transformers.MusicXTransformer(
        atten_strategy = args.atten_strategy,
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rotary_pos_emb=train_args["rel_pos_emb"],
        use_abs_pos_emb=train_args["abs_pos_emb"],
        emb_dropout=train_args["dropout"],
        attn_dropout=train_args["dropout"],
        ff_dropout=train_args["dropout"],
    ).to(device)
    



    # Load the checkpoint
    checkpoint_dir = args.out_dir / "checkpoints_enc123"
    if args.model_steps is None:
        checkpoint_filename = checkpoint_dir / "best_model_enc123.pt"
    else:
        checkpoint_filename = checkpoint_dir / f"model_enc123_{args.model_steps}.pt"
        
    state_dict = torch.load(checkpoint_filename, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
	    name = k[7:]
	    new_state_dict[name]  =v 

    model.load_state_dict(new_state_dict)
    logging.info(f"Loaded the model weights from: {checkpoint_filename}")
    model.eval()
    # Load the checkpoint
    # if args.model_steps is None:
    #     checkpoint_filename ='/work100/weixp/mmt3-final/mtmt/exp/test_enc123_sod-bar/checkpoints_enc123/best_model_enc123.pt' #checkpoint_dir / "best_model_enc1.pt"
    # else:
    #     checkpoint_filename = '/work100/weixp/mmt3-final/mtmt/exp/test_enc123_sod-bar/checkpoints_enc123/best_model_enc123.pt'#checkpoint_dir / f"model_{args.model_steps}.pt"
    # model.load_state_dict(torch.load(checkpoint_filename, map_location=device))
    # logging.info(f"Loaded the model weights from: {checkpoint_filename}")
    # model.eval()

    # Get special tokens
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]
    beat_0 = encoding["beat_code_map"][0]
    beat_4 = encoding["beat_code_map"][4]
    beat_16 = encoding["beat_code_map"][16]
    n_samples = len(test_loader)
    # Iterate over the dataset
    with torch.no_grad():
        data_iter = iter(test_loader)
        for i in tqdm.tqdm(range(n_samples), ncols=80):
            batch = next(data_iter)

            # ------------
            # Ground truth
            # ------------
            truth_np = batch["seq"][0].numpy()
            save_result(f"{i}_truth", truth_np, sample_dir, encoding,1)


           # -------------------
            # instrumtns-beat continuation
            # -------------------

            # Get output start tokens
            batch_input_4 = get_instrument_beat(batch)
            tgt_start = batch_input_4.to(device)
            print(tgt_start)
            if tgt_start.shape[1]<=3:
                        continue
            # # Generate new samples
            generated = model.generate(
                tgt_start,
                args.seq_len,
                eos_token=eos,
                temperature=args.temperature,
                filter_logits_fn=args.filter,
                filter_thres=args.filter_threshold,
                monotonicity_dim=("type", "beat"),
            )
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            # Save the results
            save_result(
                f"{i}_instrument-continuation",
                generated_np[0],
                sample_dir,
                encoding,
                5
            )



            # -------------------
            # 4-beat continuation
            # -------------------

            # Get output start tokens
            batch_input_4 = get_cut_beat(4,batch)
            tgt_start = batch_input_4.to(device)

            # Generate new samples
            generated = model.generate(
                tgt_start,
                args.seq_len,
                eos_token=eos,
                temperature=args.temperature,
                filter_logits_fn=args.filter,
                filter_thres=args.filter_threshold,
                monotonicity_dim=("type", "beat"),
            )
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            # Save the results
            save_result(
                f"{i}_4-beat-continuation",
                generated_np[0],
                sample_dir,
                encoding,
                2
            )

            # --------------------
            # 16-beat continuation
            # --------------------

            # Get output start tokens
            batch_input_16 = get_cut_beat(16,batch)
            tgt_start = batch_input_16.to(device)

            # Generate new samples
            generated = model.generate(
                tgt_start,
                args.seq_len,
                eos_token=eos,
                temperature=args.temperature,
                filter_logits_fn=args.filter,
                filter_thres=args.filter_threshold,
                monotonicity_dim=("type", "beat"),
            )
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            # Save results
            save_result(
                f"{i}_16-beat-continuation",
                generated_np[0],
                sample_dir,
                encoding,
                3
            )
            
            # ------------------------
            # Unconditioned generation
            # ------------------------

            # Get output start tokens
            tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            tgt_start[:, 0, 0] = sos

            # Generate new samples
            generated = model.generate(
                tgt_start,
                args.seq_len,
                eos_token=eos,
                temperature=args.temperature,
                filter_logits_fn=args.filter,
                filter_thres=args.filter_threshold,
                monotonicity_dim=("type", "beat"),
            )
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            # Save the results
            save_result(
                f"{i}_unconditioned", generated_np[0], sample_dir, encoding,4
            )


if __name__ == "__main__":
    main()
