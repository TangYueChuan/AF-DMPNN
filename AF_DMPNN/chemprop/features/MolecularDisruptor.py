from rdkit import Chem
from rdkit.Chem import BRICS, ChemicalFeatures, Draw, AllChem, Descriptors, GraphDescriptors
from typing import List, Tuple
from rdkit.Chem.Draw.rdMolDraw2D import MolDrawOptions
from rdkit.Chem.Scaffolds import MurckoScaffold

RECAP_RULES = {
    1: {'name': '胺键', 'smarts': '[C;!$(C=O)]-[N;!$(N=O);!$(N#N)]'},
    2: {'name': '酰胺键', 'smarts': 'C(=O)-N'},
    3: {'name': '酯键', 'smarts': 'C(=O)-[O;!$(O=C)]'},
    4: {'name': '脲键', 'smarts': 'N-C(=O)-N'},
    5: {'name': '醚键',
        'smarts': '[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]-O-[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]'},
    6: {'name': '烯烃', 'smarts': 'C=C'},
    7: {'name': '季碳中心', 'smarts': '[C;D4;!$(C(F)(F)F)]'},
    8: {'name': '芳香族 C-N 键', 'smarts': 'c-N'},
    9: {'name': '磺酰胺键', 'smarts': 'S(=O)(=O)-N'},
    10: {'name': '芳香族 C-O 键', 'smarts': 'c-O-[C;!$(C=O)]'},
    11: {'name': '芳香族 C-S 键', 'smarts': 'c-s'},
}


def fragment_with_brics(mol: Chem.Mol) -> List[Chem.Mol]:
    """BRICS"""
    fragmented_mol = BRICS.BreakBRICSBonds(mol)
    frag_mols = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=True)
    return frag_mols


def fragment_with_murcko(mol: Chem.Mol) -> List[Chem.Mol]:
    """murcko"""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    fragmented_mol = AllChem.DeleteSubstructs(mol, scaffold)
    # 获取片段
    fragmented_mol = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=True)
    frag_mols = fragmented_mol + (scaffold,)
    return frag_mols


fdef_path = r"D:\pycharmProject\AF_DMPNN\chemprop\features\BaseFeatures.fdef"
fact = ChemicalFeatures.BuildFeatureFactory(fdef_path)


def fragment_with_base(mol: Chem.Mol) -> List[Chem.Mol]:
    """BaseFeatures"""
    feats = fact.GetFeaturesForMol(mol)
    # draw_mol_with_atom_numbers(mol)
    matches = []
    for feat in feats:
        atom_ids = tuple(feat.GetAtomIds())  
        if len(atom_ids) <= 1:
            continue
        matches.append(atom_ids)
    fragsMol = custom_fragmentation_by_bonds(mol, matches)
    return fragsMol


def fragment_with_recap(mol: Chem.Mol) -> List[Chem.Mol]:
    """recap"""
    patterns = []
    for rule_id, rule_info in RECAP_RULES.items():
        patterns.append(Chem.MolFromSmarts(rule_info['smarts']))
    matches = []
    for pattern in patterns:
        m = mol.GetSubstructMatches(pattern)
        if m:
            matches.extend(m)
    fragsMol = custom_fragmentation_by_bonds(mol, matches)
    return fragsMol


def substructure_match_and_fragment(mol: Chem.Mol, fragmentation_method: str = 'custom'):
    """
    Args:
        mol_smiles: SMILES
        pattern_smarts: SMARTS
        fragmentation_method:  ('recap', 'brics', 'BaseFeatures','murcko','custom' ,或 'combo')

    Returns:
    """

    mol_with_mapping = Chem.Mol(mol)
    for atom in mol_with_mapping.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    if fragmentation_method == 'brics':
        fragments = fragment_with_brics(mol_with_mapping)
    elif fragmentation_method == 'murcko':
        fragments = fragment_with_murcko(mol_with_mapping)
    elif fragmentation_method == 'BaseFeatures':
        fragments = fragment_with_base(mol_with_mapping)
    elif fragmentation_method == 'recap':
        fragments = fragment_with_recap(mol_with_mapping)
    elif fragmentation_method == 'combo':
        fragments = fragment_with_brics(mol_with_mapping)
        if len(fragments)==1:
            fragments = fragment_with_murcko(mol_with_mapping)
            if len(fragments)==1:
                fragments = fragment_with_base(mol_with_mapping)
    return getFragmentInfo(mol_with_mapping, fragments)


def custom_fragmentation_by_bonds(mol: Chem.Mol,
                                  matches: List[Tuple[int]], ) -> List[Chem.Mol]:
    """
    """
    bonds_to_cut = set()

    for match in matches:
        pattern_atoms = set(match)
        for atom_idx in match:
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in pattern_atoms:
                    bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                    if bond and not bond.IsInRing():  
                        bonds_to_cut.add(bond.GetIdx())
    fragments = []
    if bonds_to_cut:
        fragmented_mol = Chem.FragmentOnBonds(mol, list(bonds_to_cut))
        fragments = Chem.GetMolFrags(fragmented_mol, asMols=True)
    if len(fragments) == 0:
        fragments.append(mol)
    return list(fragments)


def getFragmentInfo(mol, frag_mols):
    results = []
    for i, frag in enumerate(frag_mols):
        atom_info = []
        for atom in frag.GetAtoms():
            map_num = atom.GetAtomMapNum()
            original_idx = map_num - 1 if map_num > 0 else None
            atom_info.append({
                'fragment_atom_idx': atom.GetIdx(),
                'original_atom_idx': original_idx,
                'symbol': atom.GetSymbol(),
                'atom_map_num': map_num,
                'is_dummy': atom.GetAtomicNum() == 0  
            })
        valid_original_indices = [info['original_atom_idx'] for info in atom_info
                                  if info['original_atom_idx'] is not None]
        bond_info = []
        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() == 0:
                receptor_idx = atom.GetNeighbors()[0].GetAtomMapNum() - 1
                receptor_neiber = [Neighbor.GetIdx() for Neighbor in
                                   mol.GetAtomWithIdx(receptor_idx).GetNeighbors()]
                receptor_neiber.extend(valid_original_indices)
                if list(set(receptor_neiber) - set(valid_original_indices)):
                    ligand_idx = list(set(receptor_neiber) - set(valid_original_indices))[0]
                else:
                    ligand_idx = 0
                bond_info.append({
                    'receptor_idx': receptor_idx,
                    'ligand_idx': ligand_idx,
                    "ligand_group_idx": 0
                })

        results.append({
            'fragment_index': i,
            'mol': frag,
            'smiles': Chem.MolToSmiles(frag),
            'atom_info': atom_info,
            'bond_info': bond_info,
            'original_atom_indices': valid_original_indices,
            'num_atoms': len(valid_original_indices)
        })
    for group in results:
        for bond_info in group["bond_info"]:
            ligand_group_idx = [group["fragment_index"] for group in results if
                                bond_info["ligand_idx"] in group["original_atom_indices"]]
            if len(ligand_group_idx) == 1:
                bond_info["ligand_group_idx"] = ligand_group_idx[0]
            else:
                raise Exception()
    return results


def draw_mol_with_atom_numbers(smiles_or_mol, highlight_atoms=None):
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else:
        mol = smiles_or_mol

    if mol is None:
        print("invalid SMILES")
        return None
    opts = MolDrawOptions()
    opts.addAtomIndices = True  
    opts.atomLabelDeuteriumTritium = True  

    img = Draw.MolToImage(mol, size=(400, 300), options=opts,
                          highlightAtoms=highlight_atoms)
    img.save("img.png")


def calculate_fraction_csp3(mol):
    """"""
    carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
    if not carbon_atoms:
        return 0.0

    sp3_carbons = 0
    for atom in carbon_atoms:
        hybridization = atom.GetHybridization()
        if hybridization == Chem.HybridizationType.SP3:
            sp3_carbons += 1

    return sp3_carbons / len(carbon_atoms)

def getFragFeature(fragment):
    features = []
    features.append(fragment.GetNumAtoms())  
    features.append(fragment.GetNumHeavyAtoms())  
    features.append(Descriptors.MolWt(fragment))  
    features.append(calculate_fraction_csp3(fragment))  
    features.append(Descriptors.MolLogP(fragment))  
    features.append(Descriptors.TPSA(fragment))  
    features.append(Descriptors.MolMR(fragment)) 
    features.append(Descriptors.NumHDonors(fragment))  
    features.append(Descriptors.NumHAcceptors(fragment))  
    features.append(abs(Descriptors.MaxPartialCharge(fragment)))  
    features.append(abs(Descriptors.MinPartialCharge(fragment)))  
    topo_features = []

    topo_features.append(GraphDescriptors.BertzCT(fragment)) 
    topo_features.append(GraphDescriptors.Chi0v(fragment))  
    topo_features.append(GraphDescriptors.Kappa1(fragment))  

    topo_features.append(GraphDescriptors.HallKierAlpha(fragment)) 

    topo_features.append(Descriptors.RingCount(fragment)) 
    topo_features.append(Descriptors.NumAromaticRings(fragment))  
    topo_features.append(Descriptors.NumAliphaticRings(fragment)) 
    radical_count = sum(1 for atom in fragment.GetAtoms() if atom.GetNumRadicalElectrons() > 0)
    return features


if __name__ == '__main__':
    mol = Chem.MolFromSmiles("c1ccc(c(c1)CC(=O)[O-])Nc2c(cccc2Cl)Cl")
    # ('recap', 'brics', 'base','murcko' ,or 'custom')
    draw_mol_with_atom_numbers(mol)
    frags = substructure_match_and_fragment(mol, "brics")
    for frag in frags:
        print(frag["smiles"])
        # getFragFeature(frag["mol"])
        # print("**********")
