from torch_geometric.loader import DataLoader as PygDataLoader
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import ConcatDataset
from chordgnn.data.datasets import (
    MCMAGraphPGVoiceSeparationDataset,
    Bach370ChoralesPGVoiceSeparationDataset,
    HaydnStringQuartetPGVoiceSeparationDataset,
    MCMAGraphVoiceSeparationDataset,
    Bach370ChoralesGraphVoiceSeparationDataset,
    HaydnStringQuartetGraphVoiceSeparationDataset,
    MozartStringQuartetGraphVoiceSeparationDataset,
    MozartStringQuartetPGGraphVoiceSeparationDataset,
    CrimGraphPGVoiceSeparationDataset,
    AugmentedNetChordGraphDataset,
    Augmented2022ChordGraphDataset,
)
from torch.nn import functional as F
from collections import defaultdict
from sklearn.model_selection import train_test_split
from chordgnn.utils import add_reverse_edges_from_edge_index
from chordgnn.data.samplers import BySequenceLengthSampler
import numpy as np


class GraphPGMixVSDataModule(LightningDataModule):
    def __init__(
        self, batch_size=1, num_workers=4, force_reload=False, test_collections=None, pot_edges_dist = 2
    ):
        super(GraphPGMixVSDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.datasets = [
            # CrimGraphPGVoiceSeparationDataset(
            #     force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist = pot_edges_dist
            # ),
            Bach370ChoralesPGVoiceSeparationDataset(
                force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist = pot_edges_dist
            ),
            MCMAGraphPGVoiceSeparationDataset(
                force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_dist
            ),
            # HaydnStringQuartetPGVoiceSeparationDataset(
            #     force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_dist
            # ),
            # MozartStringQuartetPGGraphVoiceSeparationDataset(
            #     force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_dist
            # )

        ]
        if not (all([d.features == self.datasets[0].features for d in self.datasets])):
            raise Exception("Input dataset has different features")
        self.features = self.datasets[0].features
        self.test_collections = test_collections

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.datasets_map = [(dataset_i,piece_i) for dataset_i, dataset in enumerate(self.datasets) for piece_i in range(len(dataset))]
        if self.test_collections is None:
            idxs = range(len(self.datasets_map))
            collections = [self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection for i in idxs]
            trainval_idx, test_idx = train_test_split(idxs, test_size=0.3, stratify=collections, random_state=0)
            trainval_collections = [collections[i] for i in trainval_idx]
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections, random_state=0)

            # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
            test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
            train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
            val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)

            # create the datasets
            self.dataset_train = ConcatDataset([self.datasets[k][train_idx_dict[k]] for k in train_idx_dict.keys()])
            self.dataset_val = ConcatDataset([self.datasets[k][val_idx_dict[k]] for k in val_idx_dict.keys()])
            self.dataset_test = ConcatDataset([self.datasets[k][test_idx_dict[k]] for k in test_idx_dict.keys()])
            print("Running on all collections")
            print(
                f"Train size :{len(self.dataset_train)}, Val size :{len(self.dataset_val)}, Test size :{len(self.dataset_test)}"
            )
        else:
            # idxs = torch.randperm(len(self.datasets_map)).long()
            idxs = range(len(self.datasets_map))
            test_idx = [
                i
                for i in idxs
                if self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection in self.test_collections
            ]
            trainval_idx = [i for i in idxs if i not in test_idx]
            trainval_collections = [self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection for i in trainval_idx]
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections, random_state=0)
            # nidx = int(len(trainval_idx) * 0.9)
            # train_idx = trainval_idx[:nidx]
            # val_idx = trainval_idx[nidx:]

            # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
            test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
            train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
            val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)

            # create the datasets
            self.dataset_train = ConcatDataset([self.datasets[k][train_idx_dict[k]] for k in train_idx_dict.keys()])
            self.dataset_val = ConcatDataset([self.datasets[k][val_idx_dict[k]] for k in val_idx_dict.keys()])
            self.dataset_test = ConcatDataset([self.datasets[k][test_idx_dict[k]] for k in test_idx_dict.keys()])
            print(f"Running evaluation on collections {self.test_collections}")
            print(
                f"Train size :{len(self.dataset_train)}, Val size :{len(self.dataset_val)}, Test size :{len(self.dataset_test)}"
            )
        # compute the ratio between real edges and potential edges
        # real_pot_ratios = list()
        # self.real_pot_ratio = sum([graph["truth_edges_mask"].shape[0]/torch.sum(graph["truth_edges_mask"]) for dataset in self.datasets for graph in dataset.graphs])/len(self.datasets_map)
        self.pot_real_ratio = sum([d.get_positive_weight() for d in self.datasets])/len(self.datasets)

    def train_dataloader(self):
        return PygDataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return PygDataLoader(
            self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return PygDataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    def num_dropped_truth_edges(self):
        return sum([d.num_dropped_truth_edges() for d in self.datasets])




def idx_tuple_to_dict(idx_tuple, datasets_map):
    """Transforms indices of a list of tuples of indices (dataset, piece_in_dataset) 
    into a dict {dataset: [piece_in_dataset,...,piece_in_dataset]}"""
    result_dict = defaultdict(list)
    for x in idx_tuple:
        result_dict[datasets_map[x][0]].append(datasets_map[x][1])
    return result_dict


class GraphMixVSDataModule(LightningDataModule):
    def __init__(
            self, batch_size=1, num_workers=4, force_reload=False, test_collections=None, pot_edges_max_dist=2
    ):
        super(GraphMixVSDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.normalize_features = True
        self.datasets = [
            Bach370ChoralesGraphVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_max_dist),
            MCMAGraphVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_max_dist),
            HaydnStringQuartetGraphVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_max_dist),
            MozartStringQuartetGraphVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_max_dist),
        ]
        if not (all([d.features == self.datasets[0].features for d in self.datasets])):
            raise Exception("Input dataset has different features, Datasets {} with sizes: {}".format(
                " ".join([d.name for d in self.datasets]), " ".join([str(d.features) for d in self.datasets])))
        self.features = self.datasets[0].features
        self.test_collections = test_collections

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.datasets_map = [(dataset_i, piece_i) for dataset_i, dataset in enumerate(self.datasets) for piece_i in
                             range(len(dataset))]
        if self.test_collections is None:
            idxs = range(len(self.datasets_map))
            collections = [self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection for i
                           in idxs]
            trainval_idx, test_idx = train_test_split(idxs, test_size=0.3, stratify=collections, random_state=0)
            trainval_collections = [collections[i] for i in trainval_idx]
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections,
                                                  random_state=0)

            # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
            test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
            train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
            val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)

            # create the datasets
            self.dataset_train = ConcatDataset([self.datasets[k][train_idx_dict[k]] for k in train_idx_dict.keys()])
            self.dataset_val = ConcatDataset([self.datasets[k][val_idx_dict[k]] for k in val_idx_dict.keys()])
            self.dataset_test = ConcatDataset([self.datasets[k][test_idx_dict[k]] for k in test_idx_dict.keys()])
            print("Running on all collections")
            print(
                f"Train size :{len(self.dataset_train)}, Val size :{len(self.dataset_val)}, Test size :{len(self.dataset_test)}"
            )
        else:
            # idxs = torch.randperm(len(self.datasets_map)).long()
            idxs = range(len(self.datasets_map))
            test_idx = [
                i
                for i in idxs
                if self.datasets[self.datasets_map[i][0]].graphs[
                       self.datasets_map[i][1]].collection in self.test_collections
            ]
            trainval_idx = [i for i in idxs if i not in test_idx]
            trainval_collections = [
                self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection for i in
                trainval_idx]
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections,
                                                  random_state=0)
            # nidx = int(len(trainval_idx) * 0.9)
            # train_idx = trainval_idx[:nidx]
            # val_idx = trainval_idx[nidx:]

            # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
            test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
            train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
            val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)

            # create the datasets
            self.dataset_train = ConcatDataset([self.datasets[k][train_idx_dict[k]] for k in train_idx_dict.keys()])
            self.dataset_val = ConcatDataset([self.datasets[k][val_idx_dict[k]] for k in val_idx_dict.keys()])
            self.dataset_test = ConcatDataset([self.datasets[k][test_idx_dict[k]] for k in test_idx_dict.keys()])
            print(f"Running evaluation on collections {self.test_collections}")
            print(
                f"Train size :{len(self.dataset_train)}, Val size :{len(self.dataset_val)}, Test size :{len(self.dataset_test)}"
            )

    def collate_fn(self, batch):
        batch_inputs, edges, batch_label, edge_type, pot_edges, truth_edges, na, name = batch[0]
        batch_inputs = F.normalize(batch_inputs.squeeze(0)) if self.normalize_features else batch_inputs.squeeze(0)
        batch_label = batch_label.squeeze(0)
        edges = edges.squeeze(0)
        edge_type = edge_type.squeeze(0)
        pot_edges = pot_edges.squeeze(0)
        truth_edges = torch.tensor(truth_edges.squeeze()).to(pot_edges.device)
        na = torch.tensor(na)
        return batch_inputs, edges, batch_label, edge_type, pot_edges, truth_edges, na, name[0]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn
        )


class AugmentedGraphDatamodule(LightningDataModule):
    def __init__(self, batch_size=1, num_workers=4, force_reload=False, include_synth=False, num_tasks=11, collection="all", version="v1.0.0"):
        super(AugmentedGraphDatamodule, self).__init__()
        self.bucket_boundaries = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.normalize_features = True
        self.version = version
        data_source = AugmentedNetChordGraphDataset(
            force_reload=self.force_reload, nprocs=self.num_workers,
            include_synth=include_synth, num_tasks=num_tasks, collection=collection
        ) if version=="v1.0.0" else Augmented2022ChordGraphDataset(
                    force_reload=self.force_reload, nprocs=self.num_workers,
                    include_synth=include_synth, num_tasks=num_tasks, collection=collection)
        self.datasets = [data_source]
        self.tasks = self.datasets[0].tasks
        if not (all([d.features == self.datasets[0].features for d in self.datasets])):
            raise Exception("Input dataset has different features, Datasets {} with sizes: {}".format(
                " ".join([d.name for d in self.datasets]), " ".join([str(d.features) for d in self.datasets])))
        self.features = self.datasets[0].features

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.datasets_map = [(dataset_i, piece_i) for dataset_i, dataset in enumerate(self.datasets) for piece_i in
                             range(len(dataset))]

        idxs = range(len(self.datasets_map))

        test_idx = [
            i
            for i in idxs
            if self.datasets[self.datasets_map[i][0]].graphs[
                   self.datasets_map[i][1]].collection == "test"
        ]

        # val_idx = [
        #     i
        #     for i in idxs
        #     if self.datasets[self.datasets_map[i][0]].graphs[
        #            self.datasets_map[i][1]].collection == "validation"
        # ]

        train_idx = [
            i
            for i in idxs
            if self.datasets[self.datasets_map[i][0]].graphs[
                   self.datasets_map[i][1]].collection == "training"
        ]

        # trainval_idx, test_idx = train_test_split(idxs, test_size=0.3, stratify=None, random_state=0)
        # trainval_collections = [collections[i] for i in trainval_idx]
        # train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=None, random_state=0)

        # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
        test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
        train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
        # val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)

        # create the datasets
        self.dataset_train = ConcatDataset([self.datasets[k][train_idx_dict[k]] for k in train_idx_dict.keys()])
        # self.dataset_val = ConcatDataset([self.datasets[k][val_idx_dict[k]] for k in val_idx_dict.keys()])
        self.dataset_test = ConcatDataset([self.datasets[k][test_idx_dict[k]] for k in test_idx_dict.keys()])
        print("Running on all collections")
        print(
            f"Train size :{len(self.dataset_train)}, Val size :{len(self.dataset_test)}, Test size :{len(self.dataset_test)}"
        )

    def collate_fn(self, batch):
        batch_inputs, edges, edge_type, batch_label, onset_div, name = batch[0]
        # batch_inputs = F.normalize(batch_inputs.squeeze(0)) if self.normalize_features else batch_inputs.squeeze(0)
        batch_inputs = batch_inputs.squeeze(0).float()
        batch_labels = batch_label.squeeze(0)
        onset_div = onset_div.squeeze().to(batch_inputs.device)
        if self.version == "v1.0.0":
            from chordgnn.utils.chord_representations import available_representations
        else:
            from chordgnn.utils.chord_representations_latest import available_representations
        batch_label = {task: batch_labels[:, i].squeeze().long() for i, task in enumerate(available_representations.keys())}
        batch_label["onset"] = batch_labels[:, -1].squeeze()
        edges = edges.squeeze(0)
        # Add reverse edges
        edge_type = edge_type.squeeze(0)
        edges, edge_type = add_reverse_edges_from_edge_index(edges, edge_type)
        # edges = torch.cat([edges, edges.flip(0)], dim=1)
        # edge_type = torch.cat([edge_type, edge_type], dim=0)
        return batch_inputs, edges, edge_type, batch_label, onset_div, name

    def collate_train_fn(self, examples):
        lengths = list()
        x = list()
        edge_index = list()
        edge_types = list()
        y = list()
        onset_divs = list()
        max_idx = []
        max_onset_div = []
        for e in examples:
            lengths.append(e[3].shape[0])
            x.append(e[0])
            edge_index.append(e[1])
            edge_types.append(e[2])
            y.append(e[3])
            onset_divs.append(e[4])
            max_idx.append(e[0].shape[0])
            max_onset_div.append(e[4].max().item() + 1)
        lengths = torch.tensor(lengths).long()
        lengths, perm_idx = lengths.sort(descending=True)
        perm_idx = perm_idx.tolist()
        max_idx = np.cumsum(np.array([0] + [max_idx[i] for i in perm_idx]))
        max_onset_div = np.cumsum(np.array([0] + [max_onset_div[i] for i in perm_idx]))
        x = torch.cat([x[i] for i in perm_idx], dim=0).float()
        edge_index = torch.cat([edge_index[pi]+max_idx[i] for i, pi in enumerate(perm_idx)], dim=1).long()
        edge_types = torch.cat([edge_types[i] for i in perm_idx], dim=0).long()
        # y = torch.cat([y[i] for i in perm_idx], dim=0).float()
        # batch_label = {task: y[:, i].squeeze().long() for i, task in
        #                enumerate(available_representations.keys())}
        # batch_label["onset"] = y[:, -1]
        y = torch.nn.utils.rnn.pad_sequence([y[i] for i in perm_idx], batch_first=True, padding_value=-1)
        if self.version == "v1.0.0":
            from chordgnn.utils.chord_representations import available_representations
        else:
            from chordgnn.utils.chord_representations_latest import available_representations
        batch_label = {task: y[:, :, i].squeeze().long() for i, task in
                       enumerate(available_representations.keys())}
        batch_label["onset"] = y[:, :, -1]
        onset_divs = torch.cat([onset_divs[pi]+max_onset_div[i] for i, pi in enumerate(perm_idx)], dim=0).long()
        return x, edge_index, edge_types, batch_label, onset_divs, lengths

    def train_dataloader(self):
        sampler = BySequenceLengthSampler(self.dataset_train, self.bucket_boundaries, self.batch_size)
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_sampler=sampler,
            batch_size=1,
            num_workers=0,
            collate_fn=self.collate_train_fn,
            drop_last=False,
            pin_memory=False,
        )

    def val_dataloader(self):
        # batch_size = len(self.dataset_test)//10
        # sampler = BySequenceLengthSampler(self.dataset_train, self.bucket_boundaries, batch_size)
        return torch.utils.data.DataLoader(
            self.dataset_test, batch_size=1, num_workers=self.num_workers, collate_fn=self.collate_fn,
            drop_last=False, pin_memory=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test, batch_size=1, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn
        )




