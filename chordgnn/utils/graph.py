import os
import random, string
import pickle
from chordgnn.utils.general import exit_after
from chordgnn.descriptors.general import *
import torch
from numpy.lib import recfunctions as rfn


class ScoreGraph(object):
    def __init__(
        self,
        note_features,
        edges,
        name=None,
        note_array=None,
        edge_weights=None,
        labels=None,
        mask=None,
        info={},
    ):
        self.node_features = note_array.dtype.names if note_array.dtype.names else []
        self.features = note_features
        # Filter out string fields of structured array.
        if self.node_features:
            self.node_features = [
                feat
                for feat in self.node_features
                if note_features.dtype.fields[feat][0] != np.dtype("U256")
            ]
            self.features = self.features[self.node_features]
        self.x = torch.from_numpy(
            np.asarray(
                rfn.structured_to_unstructured(self.features)
                if self.node_features
                else self.features
            )
        )
        self.note_array = note_array
        self.edge_index = torch.from_numpy(edges).long()
        self.edge_weights = (
            edge_weights if edge_weights is None else torch.from_numpy(edge_weights)
        )
        self.name = name
        self.mask = mask
        self.info = info
        self.y = labels if labels is None else torch.from_numpy(labels)

    def adj(self):
        # ones = np.ones(len(self.edge_index[0]), np.uint32)
        # matrix = sp.coo_matrix((ones, (self.edge_index[0], self.edge_index[1])))
        ones = torch.ones(len(self.edge_index[0]))
        matrix = torch.sparse_coo_tensor(
            self.edge_index, ones, (len(self.x), len(self.x))
        )
        return matrix

    def save(self, save_dir):
        save_name = (
            self.name
            if self.name
            else "".join(random.choice(string.ascii_letters) for i in range(10))
        )
        (
            os.makedirs(os.path.join(save_dir, save_name))
            if not os.path.exists(os.path.join(save_dir, save_name))
            else None
        )
        with open(os.path.join(save_dir, save_name, "x.npy"), "wb") as f:
            np.save(f, self.x.numpy())
        with open(os.path.join(save_dir, save_name, "edge_index.npy"), "wb") as f:
            np.save(f, self.edge_index.numpy())
        if isinstance(self.y, torch.Tensor):
            with open(os.path.join(save_dir, save_name, "y.npy"), "wb") as f:
                np.save(f, self.y.numpy())
        if isinstance(self.edge_weights, torch.Tensor):
            np.save(
                open(os.path.join(save_dir, save_name, "edge_weights.npy"), "wb"),
                self.edge_weights.numpy(),
            )
        if isinstance(self.note_array, np.ndarray):
            np.save(
                open(os.path.join(save_dir, save_name, "note_array.npy"), "wb"),
                self.note_array,
            )
        with open(os.path.join(save_dir, save_name, "graph_info.pkl"), "wb") as handle:
            pickle.dump(
                {
                    "node_features": self.node_features,
                    "mask": self.mask,
                    "info": self.info,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


@exit_after(10)
def load_score_graph(load_dir, name=None):
    path = (
        os.path.join(load_dir, name) if os.path.basename(load_dir) != name else load_dir
    )
    if not os.path.exists(path) or not os.path.isdir(path):
        raise ValueError("The directory is not recognized.")
    x = np.load(open(os.path.join(path, "x.npy"), "rb"))
    edge_index = np.load(open(os.path.join(path, "edge_index.npy"), "rb"))
    graph_info = pickle.load(open(os.path.join(path, "graph_info.pkl"), "rb"))
    y = (
        np.load(open(os.path.join(path, "y.npy"), "rb"))
        if os.path.exists(os.path.join(path, "y.npy"))
        else None
    )
    edge_weights = (
        np.load(open(os.path.join(path, "edge_weights.npy"), "rb"))
        if os.path.exists(os.path.join(path, "edge_weights.npy"))
        else None
    )
    name = name if name else os.path.basename(path)
    return ScoreGraph(
        note_array=x,
        edges=edge_index,
        name=name,
        labels=y,
        edge_weights=edge_weights,
        mask=graph_info["mask"],
        info=graph_info["info"],
    )


def check_note_array(na):
    dtypes = na.dtype.names
    if not all(
        [
            x in dtypes
            for x in ["onset_beat", "duration_beat", "ts_beats", "ts_beat_type"]
        ]
    ):
        raise (TypeError("The given Note array is missing necessary fields."))


def graph_from_note_array(note_array, rest_array=None, norm2bar=True):
    """Turn note_array to homogeneous graph dictionary.

    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    rest_array : structured array
        A structured rest array similar to the note array but for rests.
    """

    edg_src = list()
    edg_dst = list()
    start_rest_index = len(note_array)
    for i, x in enumerate(note_array):
        for j in np.where(
            (
                np.isclose(
                    note_array["onset_beat"], x["onset_beat"], rtol=1e-04, atol=1e-04
                )
                == True
            )
            & (note_array["pitch"] != x["pitch"])
        )[0]:
            edg_src.append(i)
            edg_dst.append(j)

        for j in np.where(
            np.isclose(
                note_array["onset_beat"],
                x["onset_beat"] + x["duration_beat"],
                rtol=1e-04,
                atol=1e-04,
            )
            == True
        )[0]:
            edg_src.append(i)
            edg_dst.append(j)

        if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
            for j in np.where(
                np.isclose(
                    rest_array["onset_beat"],
                    x["onset_beat"] + x["duration_beat"],
                    rtol=1e-04,
                    atol=1e-04,
                )
                == True
            )[0]:
                edg_src.append(i)
                edg_dst.append(j + start_rest_index)

        for j in np.where(
            (x["onset_beat"] < note_array["onset_beat"])
            & (x["onset_beat"] + x["duration_beat"] > note_array["onset_beat"])
        )[0]:
            edg_src.append(i)
            edg_dst.append(j)

    if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
        for i, r in enumerate(rest_array):
            for j in np.where(
                np.isclose(
                    note_array["onset_beat"],
                    r["onset_beat"] + r["duration_beat"],
                    rtol=1e-04,
                    atol=1e-04,
                )
                == True
            )[0]:
                edg_src.append(start_rest_index + i)
                edg_dst.append(j)

        feature_fn = [
            dname
            for dname in note_array.dtype.names
            if dname not in rest_array.dtype.names
        ]
        if feature_fn:
            rest_feature_zeros = np.zeros((len(rest_array), len(feature_fn)))
            rest_feature_zeros = rfn.unstructured_to_structured(
                rest_feature_zeros, dtype=list(map(lambda x: (x, "<4f"), feature_fn))
            )
            rest_array = rfn.merge_arrays((rest_array, rest_feature_zeros))
    else:
        end_times = note_array["onset_beat"] + note_array["duration_beat"]
        for et in np.sort(np.unique(end_times))[:-1]:
            if et not in note_array["onset_beat"]:
                scr = np.where(end_times == et)[0]
                diffs = note_array["onset_beat"] - et
                tmp = np.where(diffs > 0, diffs, np.inf)
                dst = np.where(tmp == tmp.min())[0]
                for i in scr:
                    for j in dst:
                        edg_src.append(i)
                        edg_dst.append(j)

    edges = np.array([edg_src, edg_dst])
    # Resize Onset Beat to bar
    if norm2bar:
        note_array["onset_beat"] = np.mod(
            note_array["onset_beat"], note_array["ts_beats"]
        )
        if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
            rest_array["onset_beat"] = np.mod(
                rest_array["onset_beat"], rest_array["ts_beats"]
            )

    nodes = np.hstack((note_array, rest_array))
    return nodes, edges
