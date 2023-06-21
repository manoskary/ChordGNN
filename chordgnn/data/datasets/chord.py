import numpy as np

from chordgnn.utils import load_score_hgraph, hetero_graph_from_note_array, select_features, HeteroScoreGraph
from chordgnn.utils import time_divided_tsv_to_part
from chordgnn.models.core import positional_encoding
from chordgnn.utils.chord_representations import available_representations
import torch
import os
from chordgnn.data.dataset import BuiltinDataset, chordgnnDataset
from joblib import Parallel, delayed
from tqdm import tqdm
import random


class AugmentedNetChordDataset(BuiltinDataset):
    r"""The AugmentedNet Chord Dataset.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if AugmentedNet Chord Dataset scores are already available otherwise it will download it.
        Default: ~/.chordgnn/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, is_zip=True):
        url = "https://github.com/napulen/AugmentedNet/releases/download/v1.0.0/dataset.zip"
        super(AugmentedNetChordDataset, self).__init__(
            name="AugmentedNetChordDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def process(self, subset=""):
        self.scores = list()
        for root, dirs, files in os.walk(self.raw_path):
            if root.endswith(subset):
                for file in files:
                    if file.endswith(".tsv") and not file.startswith("dataset_summary"):
                        self.scores.append(os.path.join(root, file))

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False



class AugmentedNetLatestChordDataset(BuiltinDataset):
    r"""The AugmentedNet Chord Dataset.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if AugmentedNet Chord Dataset scores are already available otherwise it will download it.
        Default: ~/.chordgnn/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, is_zip=True):
        url = "https://github.com/napulen/AugmentedNet/releases/latest/download/dataset.zip"
        super(AugmentedNetLatestChordDataset, self).__init__(
            name="AugmentedNetLatestChordDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def process(self, subset=""):
        self.scores = list()
        for root, dirs, files in os.walk(self.raw_path):
            if root.endswith(subset):
                for file in files:
                    if file.endswith(".tsv") and not file.startswith("dataset_summary"):
                        self.scores.append(os.path.join(root, file))

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False


class ChordGraphDataset(chordgnnDataset):
    def __init__(self, dataset_base, max_size=None, verbose=True, nprocs=1, name=None, raw_dir=None, force_reload=False, prob_pieces=[]):
        self.dataset_base = dataset_base
        self.prob_pieces = prob_pieces
        self.dataset_base.process()
        self.max_size = max_size
        if verbose:
            print("Loaded AugmentedNetChordDataset Successfully, now processing...")
        self.graph_dicts = list()
        self.n_jobs = nprocs
        super(ChordGraphDataset, self).__init__(
            name=name,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def process(self):
        Parallel(self.n_jobs)(delayed(self._process_score)(fn) for fn in
                              tqdm(self.dataset_base.scores, desc="Processing AugmentedNetChordGraphDataset"))
        self.load()

    def _process_score(self, score_fn):
        pass
    def has_cache(self):
        # return True
        if all([
            os.path.exists(os.path.join(self.save_path, os.path.splitext(os.path.basename(path))[0])) for path
            in
            self.dataset_base.scores]):
            return True
        return False

    def save(self):
        """save the graph list and the labels"""
        pass

    def load(self):
        self.graphs = list()
        for fn in os.listdir(self.save_path):
            path = os.path.join(self.save_path, fn)
            graph = load_score_hgraph(path, fn)
            if not self.include_synth and graph.name.endswith("-synth"):
                continue
            if self.collection != "all" and not graph.name.startswith(
                    self.collection) and graph.collection == "test":
                continue
            if graph.name in self.prob_pieces:
                continue
            self.graphs.append(graph)

    @property
    def features(self):
        return self.graphs[0].x.shape[-1]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return [
            self.get_graph_attr(i)
            for i in idx
        ]

    def get_graph_attr(self, idx):
        if self.graphs[idx].x.size(0) > self.max_size and self.graphs[idx].collection != "test":
            random_idx = random.randint(0, self.graphs[idx].x.size(0) - self.max_size)
            indices = torch.arange(random_idx, random_idx + self.max_size)
            edge_indices = torch.isin(self.graphs[idx].edge_index[0], indices) & torch.isin(
                self.graphs[idx].edge_index[1], indices)
            onset_divs = torch.tensor(
                self.graphs[idx].note_array["onset_div"][random_idx:random_idx + self.max_size])
            unique_onsets = torch.unique(torch.tensor(self.graphs[idx].note_array["onset_div"]), sorted=True)
            label_idx = (unique_onsets >= onset_divs.min()) & (unique_onsets <= onset_divs.max())
            return [
                self.graphs[idx].x[indices],
                self.graphs[idx].edge_index[:, edge_indices] - random_idx,
                self.graphs[idx].edge_type[edge_indices],
                self.graphs[idx].y[label_idx],
                onset_divs,
                self.graphs[idx].name
            ]

        else:
            return [
                self.graphs[idx].x,
                self.graphs[idx].edge_index,
                self.graphs[idx].edge_type,
                self.graphs[idx].y,
                torch.tensor(self.graphs[idx].note_array["onset_div"]),
                self.graphs[idx].name
            ]


class AugmentedNetChordGraphDataset(ChordGraphDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, nprocs=4, include_synth=False, num_tasks=11, collection="all", max_size=512):
        dataset_base = AugmentedNetChordDataset(raw_dir=raw_dir)
        self.collection = collection
        # Collection is one of ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern"]
        assert self.collection in ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern", "all"]
        self.include_synth = include_synth
        # Problematic Pieces
        self.prob_pieces = []# ["bps-29-op106-hammerklavier-1", "tavern-mozart-k613-b", "tavern-mozart-k613-a", "abc-op127-4", "mps-k533-1", "abc-op59-no1-1"]
        # Frog model order: key, tonicisation, degree, quality, inversion, and root
        if isinstance(num_tasks, int):
            if num_tasks <= 6:
                self.tasks = {
                    "localkey": 35, "tonkey": 35, "degree1": 22, "degree2": 22, "quality": 16, "inversion": 4, "root": 35}
            elif num_tasks == 11:
                self.tasks = {
                    "localkey": 35, "tonkey": 35, "degree1": 22, "degree2": 22, "quality": 16, "inversion": 4,
                    "root": 35, "romanNumeral": 76, "hrhythm": 2, "pcset": 94, "bass": 35,
                }
        else:
            from chordgnn.utils.chord_representations import available_representations
            self.tasks = {num_tasks: len(available_representations[num_tasks].classList)}
        super(AugmentedNetChordGraphDataset, self).__init__(
            dataset_base=dataset_base,
            max_size=max_size,
            nprocs=nprocs,
            name="AugmentedNetChordGraphDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose)

    def _process_score(self, score_fn):
        name = os.path.splitext(os.path.basename(score_fn))[0]
        name = name + "-synth" if os.path.join("AugmentedNetChordDataset", "dataset-synth") in score_fn else name
        # name = name + "-2-synth" if os.path.join("AugmentedNetChordDataset", "dataset-synth-2") in score_fn else name
        # Skip synthetic scores in testing.

        if os.path.join("AugmentedNetChordDataset", "dataset-synth") in score_fn and os.path.basename(os.path.dirname(score_fn)) in ["test"]:
            return
        collection = "training" if os.path.basename(
            os.path.dirname(score_fn)) == "validation" else os.path.basename(os.path.dirname(score_fn))
        if collection == "test":
            note_array, labels = time_divided_tsv_to_part(score_fn, transpose=False)
            data_to_graph(note_array, labels, collection, name, save_path=self.save_path)
        else:
            x = time_divided_tsv_to_part(score_fn, transpose=True)
            for i, (note_array, labels) in enumerate(x):
                data_to_graph(note_array, labels, collection, (name + "-{}".format(i) if i > 0 else name), save_path=self.save_path)
        return


class Augmented2022ChordGraphDataset(ChordGraphDataset):

    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True, nprocs=4, include_synth=False, num_tasks=11, collection="all", max_size=512):
        dataset_base = AugmentedNetLatestChordDataset(raw_dir=raw_dir)
        # Collection is one of ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern"]
        self.collection = collection
        assert self.collection in ["abc", "bps", "haydnop20", "wir", "wirwtc", "tavern", "mps", "all"]
        self.include_synth = include_synth
        # Problematic Pieces
        prob_pieces = [
            'keymodt-reger-96A',
            'keymodt-rimsky-korsakov-3-23c',
            'keymodt-reger-88A',
            'keymodt-reger-68',
            'keymodt-reger-84',
            'keymodt-rimsky-korsakov-3-24a',
            'keymodt-rimsky-korsakov-3-23b',
            'keymodt-aldwell-ex27-4b',
            'keymodt-reger-42a',
            'keymodt-reger-08',
            'mps-k545-1',
            'keymodt-tchaikovsky-173b',
            'keymodt-reger-73',
            'mps-k282-3',
            'keymodt-tchaikovsky-173j',
            'keymodt-kostka-payne-ex19-5',
            'mps-k457-3',
            'keymodt-reger-59',
            'keymodt-reger-82',
            'keymodt-rimsky-korsakov-3-5h',
            'mps-k332-2',
            'mps-k310-1',
            'mps-k457-2',
            'keymodt-rimsky-korsakov-3-7',
            'mps-k576-1',
            'keymodt-kostka-payne-ex18-4',
            'keymodt-reger-81',
            'keymodt-reger-45a',
            'keymodt-rimsky-korsakov-3-14g',
            'keymodt-reger-64',
            'keymodt-tchaikovsky-193b',
            'keymodt-reger-86A',
            'keymodt-reger-15',
            'keymodt-reger-28',
            'mps-k309-1',
            'keymodt-reger-99A',
            'keymodt-reger-55',
            'keymodt-tchaikovsky-189']

        # ["bps-29-op106-hammerklavier-1", "tavern-mozart-k613-b", "tavern-mozart-k613-a", "abc-op127-4", "mps-k533-1", "abc-op59-no1-1"]
        # Frog model order: key, tonicisation, degree, quality, inversion, and root
        if isinstance(num_tasks, int):
            if num_tasks <= 6:
                self.tasks = {
                    "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
                    "root": 35}
            elif num_tasks == 11:
                self.tasks = {
                    "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
                    "root": 35, "romanNumeral": 31, "hrhythm": 7, "pcset": 121, "bass": 35}
            elif num_tasks == 14:
                self.tasks = {
                    "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
                    "root": 35, "romanNumeral": 31, "hrhythm": 7, "pcset": 121, "bass": 35, "tenor": 35,
                    "alto": 35, "soprano": 35}
        else:
            from chordgnn.utils.chord_representations_latest import available_representations
            self.tasks = {num_tasks: len(available_representations[num_tasks].classList)}
        super(Augmented2022ChordGraphDataset, self).__init__(
            dataset_base=dataset_base,
            max_size=max_size,
            nprocs=nprocs,
            name="Augmented2022ChordGraphDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            prob_pieces=prob_pieces)

    def _process_score(self, score_fn):
        name = os.path.splitext(os.path.basename(score_fn))[0]
        name = name + "-synth" if os.path.join("AugmentedNetLatestChordDataset", "dataset-synth") in score_fn else name
        # Skip synthetic scores in testing.
        if os.path.join("AugmentedNetLatestChordDataset", "dataset-synth") in score_fn and os.path.basename(
                os.path.dirname(score_fn)) in ["test"]:
            return
        collection = "training" if os.path.basename(
            os.path.dirname(score_fn)) == "validation" else os.path.basename(os.path.dirname(score_fn))
        if collection == "test":
            note_array, labels = time_divided_tsv_to_part(score_fn, transpose=False, version="latest")
            data_to_graph(note_array, labels, collection, name, save_path=self.save_path)
        else:
            x = time_divided_tsv_to_part(score_fn, transpose=True, version="latest")
            for i, (note_array, labels) in enumerate(x):
                data_to_graph(note_array, labels, collection, (name + "-{}".format(i) if i > 0 else name),
                              save_path=self.save_path)
        return




def data_to_graph(note_array, labels, collection, name, save_path):
    nodes, edges = hetero_graph_from_note_array(note_array=note_array)
    note_features = select_features(note_array, "chord")
    hg = HeteroScoreGraph(
        note_features,
        edges,
        name=name,
        labels=labels,
        note_array=note_array,
    )
    setattr(hg, "collection", collection)
    # pos_enc = positional_encoding(hg.edge_index, len(hg.x), 20)
    # hg.x = torch.cat((hg.x, pos_enc), dim=1)
    hg.save(save_path)
    del hg, note_array, nodes, edges, note_features
    return