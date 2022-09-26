import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Model parameters")

    parser.add_argument("--seed", type=str, default='2022', help="random seed.")
    parser.add_argument("--mode", type=str, default='train', help="whether you are training a new model or using pretrained models, choose between train and test")
    parser.add_argument('--model_type', type=str, default="TCGConv", help="choose among CONCAT, ATN, TCGConv, TCGConv_sum, CGConv, CGConv_sum")
    # ========================= Data Configs ==========================
    parser.add_argument('--root', type=str, default='/workspaces/Edge-Representation-Learning-in-Temporal-Graph/')
    parser.add_argument('--graph_type', type=str, default='G', help = "Please use default graph type")
    parser.add_argument('--dataset_name', type=str, default='CC') 
    parser.add_argument('--percentage', type=float, default=1, help='percent of data used')
    parser.add_argument('--truncate_size', type=int, default=None)
    parser.add_argument('--k', type=int, default=20, help='number of k for top k')
    
    

    return parser.parse_args()