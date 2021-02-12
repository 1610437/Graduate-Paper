import csv
from pathlib import Path
import torch

def read_kansai1G():
    with open(str(Path("setdata") / "flow1/flow1G.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        population_data = torch.tensor([[int(col) for col in row] for row in reader],dtype=torch.double)

    with open(str(Path("setdata") / "flow1/neighbourG.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        adj_table = torch.tensor([[int(col) for col in row] for row in reader],dtype=torch.double)

    with open(str(Path("setdata") / "flow1/coordinateG.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        location_table = [[int(col) for col in row] for row in reader]

    with open(str(Path("./") /"theta.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        tp = torch.tensor([[float(col) for col in row] for row in reader],dtype=torch.double)

    return population_data, location_table, adj_table,tp

def read_kansai2Global():
    with open(str(Path("setdata") / "flow2/flow2G.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        population_data = torch.tensor([[int(col) for col in row] for row in reader],dtype=torch.double)

    with open(str(Path("setdata") / "flow2/neighbourG.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        adj_table = torch.tensor([[int(col) for col in row] for row in reader],dtype=torch.double)

    with open(str(Path("setdata") / "flow2/coordinateG.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        location_table = [[int(col) for col in row] for row in reader]

    with open(str(Path("./") /"theta.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        tp = torch.tensor([[float(col) for col in row] for row in reader],dtype=torch.double)

    return population_data, location_table, adj_table,tp

def read_kansai2Local():
    with open(str(Path("setdata") / "flow2/flow2L2km.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        population_data = torch.tensor([[int(col) for col in row] for row in reader],dtype=torch.double)

    with open(str(Path("setdata") / "flow2/neighbourL2km.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        adj_table = torch.tensor([[int(col) for col in row] for row in reader],dtype=torch.double)

    with open(str(Path("setdata") / "flow2/coordinateL2km.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        location_table = [[int(col) for col in row] for row in reader]

    with open(str(Path("./") /"theta.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        tp = torch.tensor([[float(col) for col in row] for row in reader],dtype=torch.double)

    return population_data, location_table, adj_table,tp
