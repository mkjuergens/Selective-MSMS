from massspecgym.data.datasets import RetrievalDataset
import numpy as np
import torch

class RetrievalDataset_PrecompFPandInchi(RetrievalDataset):
    def __init__(
        self,
        fp_pth = None,
        inchi_pth = None,
        candidates_fp_pth = None,
        candidates_inchi_pth = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metadata["fp_4096"] = list(np.load(fp_pth))
        self.metadata["inchikey"] = list(np.load(inchi_pth))
        
        self.candidate_fps = dict(np.load(candidates_fp_pth))
        self.candidate_inchi = dict(np.load(candidates_inchi_pth))
    
    def __getitem__(self, i):

        item = super(RetrievalDataset, self).__getitem__(i, transform_mol=False)

        # Save the original SMILES representation of the query molecule (for evaluation)
        item["smiles"] = item["mol"]

        # Get candidates
        if item["mol"] not in self.candidates:
            raise ValueError(f'No candidates for the query molecule {item["mol"]}.')
        item["candidates"] = self.candidates[item["mol"]]

        # Save the original SMILES representations of the canidates (for evaluation)
        item["candidates_smiles"] = item["candidates"]



        # Transform the query and candidate molecules
        item["mol"] = self.metadata["fp_4096"].iloc[i].astype(np.int32)
        item["candidates"] = self.candidate_fps[item["smiles"]]
        if isinstance(item["mol"], np.ndarray):
            item["mol"] = torch.as_tensor(item["mol"], dtype=self.dtype)
        if isinstance(item["candidates"], np.ndarray):
            item["candidates"] = torch.as_tensor(item["candidates"], dtype=self.dtype)

        item["labels"] = [
            (c == item["mol"]).all().item() for c in item["candidates"]
        ]

        if not any(item["labels"]):
            raise ValueError(
                f'Query molecule {item["mol"]} not found in the candidates list.'
            )

        return item