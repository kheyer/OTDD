from otdd import *

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

class TanimotoDistance(POTDistance):
    def __init__(self):
        super().__init__(distance_metric='jaccard')

class SmilesDataset(ArrayDataset):
    def __init__(self, smiles, labels=None, radius=3, bits=2048, multithread=True):
        self.radius = radius
        self.bits = bits
        self.smiles = smiles
        fps = self.process_smiles(smiles, multithread=multithread)
        
        super().__init__(fps, labels=labels)
        
    def get_fingerprint(self, smile):
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.bits)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp,arr)
        return fp
        
    def get_fingerprints(self, smiles, multithread=True):
        if multithread:
            p = Pool(processes=os.cpu_count())
            fps = p.map(self.get_fingerprint, smiles)
            p.close()
        else:
            fps = [self.get_fingerprint(i) for i in smiles]
            
        fps = np.array(fps, dtype=np.int8)
        
        return fps
    
    def process_smiles(self, smiles, multithread=True):
        
        if len(smiles)>100000:
            all_fps = np.zeros((len(smiles), self.bits), dtype=np.int8)
            
            for i in range(0, len(smiles), 100000):
                fps = self.get_fingerprints(smiles[i:i+100000], multithread=multithread)
                all_fps[i:i+100000] = fps
                
                del fps
                gc.collect()
                
        else:
            all_fps = self.get_fingerprints(smiles, multithread=multithread)
            
        return all_fps
    
    