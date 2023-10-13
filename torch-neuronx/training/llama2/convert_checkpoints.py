import argparse
import os

import torch
import torch_xla.utils.serialization as xser

def create_partition(num_hidden_layers, pipeline_parallel_size):
    """
    Evenly split the transformer layers between the PP ranks
    """
    assert num_hidden_layers % pipeline_parallel_size == 0
    num_layer_per_partition = num_hidden_layers  // pipeline_parallel_size
    pipeline_cuts = []
    current_cut = num_layer_per_partition - 1
    for i in range(pipeline_parallel_size-1):
        pipeline_cuts.append(f"model.layers.{current_cut}")
        current_cut += num_layer_per_partition
    print(f"pipeline_cuts {pipeline_cuts}")
    return pipeline_cuts
 
def translate_llama_full_state_dict_to_tp(full_state, tp_size, tp_rank, pp_size, pp_rank, partitions):
    partial_state = {}
    for name, full_p in full_state.items():
        ##################### PP Slice #########################################
        # Embedding only in first PP
        if pp_rank != 0 and "embed_tokens" in name:
            continue
        # LMhead and final layer norm only in last PP rank
        if pp_rank != pp_size - 1 and ("lm_head" in name or "model.norm.weight" in name):
            continue
        if "layers" in name:
            # model.layers.idx.input_layernorm.weight
            layer_idx = int(name.split(".")[2])
            # 'model.layers.idx'
            pre_layer_cut = int(partitions[pp_rank-1].split(".")[2]) if pp_rank > 0 else -10000000
            current_layer_cut = int(partitions[pp_rank].split(".")[2]) if pp_rank < pp_size-1 else 10000000
            if layer_idx <= pre_layer_cut or layer_idx > current_layer_cut:
                continue
 
        ##################### TP Slice #########################################
        if "embed_tokens" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name or "down_proj" in name or "lm_head" in name:
            # parallel embedding or ColumnParallelLinear, parallel on 0th dim
            # RowParallelLinear parallel on 1st dim
            partition_dim = 1 if ("o_proj" in name or "down_proj" in name) else 0
            dim_size = full_p.size()[partition_dim]
            assert dim_size % tp_size == 0, "vocab size is not divisiable"
            partition_size = dim_size // tp_size
            with torch.no_grad():
                to_load = full_p.narrow(partition_dim, tp_rank * partition_size, partition_size).detach().clone()
                # gc collect
                #full_p.data = torch.tensor(0.0)
                partial_state[name] = to_load
        elif "gate_proj" in name or "up_proj" in name:
            # ColumnParallelLinear
            partition_dim = 0
            dim_size = full_p.size()[partition_dim]
            assert dim_size % tp_size == 0, "vocab size is not divisiable"
            partition_size = dim_size // tp_size
            with torch.no_grad():
                to_load = full_p.narrow(partition_dim, tp_rank * partition_size, partition_size).detach().clone()
                # gc collect
                #full_p.data = torch.tensor(0.0)
            token = "gate_proj" if "gate_proj" in name else "up_proj"
            updated_name = name.replace(token, "gate_up_proj")
            if updated_name in partial_state:
                if token == "gate_proj":
                    partial_state[updated_name] = torch.cat([to_load, partial_state[updated_name]], dim=0)
                else:
                    partial_state[updated_name] = torch.cat([partial_state[updated_name], to_load], dim=0)
            else:
                partial_state[updated_name] = to_load
        else:
            # no TP
            partial_state[name] = full_p
    return partial_state

def convert_and_save(args):
    if args.convert_from_full_model:
        full_state = torch.load(args.input_dir)
    partitions = create_partition(args.n_layers, args.pp_size)
    for tp_rank in range(args.tp_size):
        for pp_rank in range(args.pp_size):
            if args.convert_from_full_model:
                partial_state = translate_llama_full_state_dict_to_tp(full_state, args.tp_size, tp_rank, args.pp_size, pp_rank, partitions)
            elif args.convert_from_xser:
                load_dir = os.path.join(args.input_dir, "tp_rank_{:02d}_pp_rank_{:02d}".format(tp_rank, pp_rank))
                partial_state = xser.load(load_dir)
            else:
                load_dir = os.path.join(args.input_dir, "tp_rank_{:02d}_pp_rank_{:02d}".format(tp_rank, pp_rank), "checkpoint.pt")
                partial_state = xser.load(load_dir)
            save_dir = os.path.join(args.output_dir, "tp_rank_{:02d}_pp_rank_{:02d}".format(tp_rank, pp_rank))
            print(f"Saving partitioned checkpoint into {save_dir}, save_xser {args.convert_to_xser}")
            if args.convert_to_xser:
                os.makedirs(args.output_dir, exist_ok=True)
                xser.save(partial_state, save_dir)
            else:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(partial_state, os.path.join(save_dir, "checkpoint.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=0)
    parser.add_argument("--convert_from_xser", action="store_true")
    parser.add_argument("--convert_to_xser", action="store_true")
    parser.add_argument("--convert_from_full_model", action="store_true")

    args, _ = parser.parse_known_args()

    convert_and_save(args)