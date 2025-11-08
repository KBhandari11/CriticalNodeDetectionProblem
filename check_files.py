
import json
import os

objectivelist=["MinMax", "CNP", "MaxNum"]
networklist=["BA", "ER", "SW", "MIX"]
networksmalllist=["ba", "er", "sw", "mixed"]
modellist=["GAT", "SAGEConv", "GCN"]

def main():
    for obj in objectivelist:
        for network,networksmall in zip(networklist,networksmalllist):
            for model in modellist:
                path = f"/gpfs/u/home/LLMG/LLMGbhnd/scratch/CNDP_data/{model}/{network}_Graph/{obj}/training_log.json"
                with open(path, "r") as f:
                    lines = f.readlines()

                episodes = []
                for line in lines:
                    line = line.strip().rstrip(",")
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if "episode" in entry:
                            episodes.append(entry["episode"])
                    except json.JSONDecodeError:
                        continue

                if not episodes:
                    print(f"\t[EMPTY] {path}")
                    continue

                max_ep = max(episodes)
                if 100000 in episodes:
                    print(f"\t[OK] {path} | Max episode: {max_ep}")
                else:
                    print(f"\t[INCOMPLETE] {path} | Max episode: {max_ep}")

if __name__ == "__main__":
    main()


#srun -t 10 ./check_file python check_files.py 