import json
import os
from tempfile import TemporaryDirectory
import pickle
from typing import List, Optional, Tuple
from typing_extensions import Literal

import torch
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

from chemprop.features import get_available_features_generators


def get_checkpoint_paths(checkpoint_path: Optional[str] = None,
                         checkpoint_paths: Optional[List[str]] = None,
                         checkpoint_dir: Optional[str] = None,
                         ext: str = '.pt') -> Optional[List[str]]:
    """
    Gets a list of checkpoint paths either from a single checkpoint path or from a directory of checkpoints.

    If :code:`checkpoint_path` is provided, only collects that one checkpoint.
    If :code:`checkpoint_paths` is provided, collects all of the provided checkpoints.
    If :code:`checkpoint_dir` is provided, walks the directory and collects all checkpoints.
    A checkpoint is any file ending in the extension ext.

    :param checkpoint_path: Path to a checkpoint.
    :param checkpoint_paths: List of paths to checkpoints.
    :param checkpoint_dir: Path to a directory containing checkpoints.
    :param ext: The extension which defines a checkpoint file.
    :return: A list of paths to checkpoints or None if no checkpoint path(s)/dir are provided.
    """
    if sum(var is not None for var in [checkpoint_dir, checkpoint_path, checkpoint_paths]) > 1:
        raise ValueError('Can only specify one of checkpoint_dir, checkpoint_path, and checkpoint_paths')

    if checkpoint_path is not None:
        return [checkpoint_path]

    if checkpoint_paths is not None:
        return checkpoint_paths

    if checkpoint_dir is not None:
        checkpoint_paths = []

        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith(ext):
                    checkpoint_paths.append(os.path.join(root, fname))

        if len(checkpoint_paths) == 0:
            raise ValueError(f'Failed to find any checkpoints with extension "{ext}" in directory "{checkpoint_dir}"')

        return checkpoint_paths

    return None


class CommonArgs(Tap):
    """:class:`CommonArgs` contains arguments that are used in both :class:`TrainArgs` and :class:`PredictArgs`."""

    smiles_column: str = "smiles"
    """Name of the column containing SMILES strings. By default, uses the first column."""
    checkpoint_dir: str = None
    """Directory from which to load model checkpoints (walks directory and ensembles all models that are found)."""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pt` file)."""
    checkpoint_paths: List[str] = None
    """List of paths to model checkpoints (:code:`.pt` files)."""
    no_cuda: bool = False
    """Turn off cuda (i.e., use CPU instead of GPU)."""
    gpu: int = None
    """Which GPU to use."""
    features_generator: List[str] = []
    """Method(s) of generating additional features."""
    features_path: List[str] = None
    """Path(s) to features to use in FNN (instead of features_generator)."""
    no_features_scaling: bool = False
    """Turn off scaling of features."""
    max_data_size: int = None
    """Maximum number of data points to load."""
    num_workers: int = 8 
    """Number of workers for the parallel data loading (0 means sequential)."""
    batch_size: int = 50
    """Batch size."""

    @property
    def device(self) -> torch.device:
        """The :code:`torch.device` on which to load and process data and models."""
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == 'cuda'
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        """Whether to use CUDA (i.e., GPUs) or not."""
        return not self.no_cuda and torch.cuda.is_available()

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda

    @property
    def features_scaling(self) -> bool:
        """Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler` to the additional molecule-level features."""
        return not self.no_features_scaling

    def add_arguments(self) -> None:
        self.add_argument('--gpu', choices=list(range(torch.cuda.device_count())))
        self.add_argument('--features_generator', choices=get_available_features_generators())

    def process_args(self) -> None:
        # Load checkpoint paths
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path=self.checkpoint_path,
            checkpoint_paths=self.checkpoint_paths,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Validate features
        if self.features_generator is not None and 'rdkit_2d_normalized' in self.features_generator and self.features_scaling:
            raise ValueError('When using rdkit_2d_normalized features, --no_features_scaling must be specified.')


class TrainArgs(CommonArgs):
    """:class:`TrainArgs` includes :class:`CommonArgs` along with additional arguments used for training a Chemprop model."""

    # General arguments
    data_path: str = r"data\clintox.csv"
    """Path to data CSV file."""
    # save_dir: str = r"model\ablation-study\Frag-DMPNN-noAtt\ESOL"
    save_dir: str = r"model\AF-DMPNN-recap\clintox"
    """Directory where model checkpoints will be saved."""
    # target_columns: List[str] = ["p_np"]
    # target_columns: List[str] = ["measured log solubility in mols per litre"]
    # target_columns: List[str] = ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER","NR-ER-LBD","NR-PPAR-gamma","SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]
    # target_columns: List[str] = ['ACEA_T47D_80hr_Negative', 'ACEA_T47D_80hr_Positive', 'APR_HepG2_CellCycleArrest_24h_dn', 'APR_HepG2_CellCycleArrest_24h_up', 'APR_HepG2_CellCycleArrest_72h_dn', 'APR_HepG2_CellLoss_24h_dn', 'APR_HepG2_CellLoss_72h_dn', 'APR_HepG2_MicrotubuleCSK_24h_dn', 'APR_HepG2_MicrotubuleCSK_24h_up', 'APR_HepG2_MicrotubuleCSK_72h_dn', 'APR_HepG2_MicrotubuleCSK_72h_up', 'APR_HepG2_MitoMass_24h_dn', 'APR_HepG2_MitoMass_24h_up', 'APR_HepG2_MitoMass_72h_dn', 'APR_HepG2_MitoMass_72h_up', 'APR_HepG2_MitoMembPot_1h_dn', 'APR_HepG2_MitoMembPot_24h_dn', 'APR_HepG2_MitoMembPot_72h_dn', 'APR_HepG2_MitoticArrest_24h_up', 'APR_HepG2_MitoticArrest_72h_up', 'APR_HepG2_NuclearSize_24h_dn', 'APR_HepG2_NuclearSize_72h_dn', 'APR_HepG2_NuclearSize_72h_up', 'APR_HepG2_OxidativeStress_24h_up', 'APR_HepG2_OxidativeStress_72h_up', 'APR_HepG2_StressKinase_1h_up', 'APR_HepG2_StressKinase_24h_up', 'APR_HepG2_StressKinase_72h_up', 'APR_HepG2_p53Act_24h_up', 'APR_HepG2_p53Act_72h_up', 'APR_Hepat_Apoptosis_24hr_up', 'APR_Hepat_Apoptosis_48hr_up', 'APR_Hepat_CellLoss_24hr_dn', 'APR_Hepat_CellLoss_48hr_dn', 'APR_Hepat_DNADamage_24hr_up', 'APR_Hepat_DNADamage_48hr_up', 'APR_Hepat_DNATexture_24hr_up', 'APR_Hepat_DNATexture_48hr_up', 'APR_Hepat_MitoFxnI_1hr_dn', 'APR_Hepat_MitoFxnI_24hr_dn', 'APR_Hepat_MitoFxnI_48hr_dn', 'APR_Hepat_NuclearSize_24hr_dn', 'APR_Hepat_NuclearSize_48hr_dn', 'APR_Hepat_Steatosis_24hr_up', 'APR_Hepat_Steatosis_48hr_up', 'ATG_AP_1_CIS_dn', 'ATG_AP_1_CIS_up', 'ATG_AP_2_CIS_dn', 'ATG_AP_2_CIS_up', 'ATG_AR_TRANS_dn', 'ATG_AR_TRANS_up', 'ATG_Ahr_CIS_dn', 'ATG_Ahr_CIS_up', 'ATG_BRE_CIS_dn', 'ATG_BRE_CIS_up', 'ATG_CAR_TRANS_dn', 'ATG_CAR_TRANS_up', 'ATG_CMV_CIS_dn', 'ATG_CMV_CIS_up', 'ATG_CRE_CIS_dn', 'ATG_CRE_CIS_up', 'ATG_C_EBP_CIS_dn', 'ATG_C_EBP_CIS_up', 'ATG_DR4_LXR_CIS_dn', 'ATG_DR4_LXR_CIS_up', 'ATG_DR5_CIS_dn', 'ATG_DR5_CIS_up', 'ATG_E2F_CIS_dn', 'ATG_E2F_CIS_up', 'ATG_EGR_CIS_up', 'ATG_ERE_CIS_dn', 'ATG_ERE_CIS_up', 'ATG_ERRa_TRANS_dn', 'ATG_ERRg_TRANS_dn', 'ATG_ERRg_TRANS_up', 'ATG_ERa_TRANS_up', 'ATG_E_Box_CIS_dn', 'ATG_E_Box_CIS_up', 'ATG_Ets_CIS_dn', 'ATG_Ets_CIS_up', 'ATG_FXR_TRANS_up', 'ATG_FoxA2_CIS_dn', 'ATG_FoxA2_CIS_up', 'ATG_FoxO_CIS_dn', 'ATG_FoxO_CIS_up', 'ATG_GAL4_TRANS_dn', 'ATG_GATA_CIS_dn', 'ATG_GATA_CIS_up', 'ATG_GLI_CIS_dn', 'ATG_GLI_CIS_up', 'ATG_GRE_CIS_dn', 'ATG_GRE_CIS_up', 'ATG_GR_TRANS_dn', 'ATG_GR_TRANS_up', 'ATG_HIF1a_CIS_dn', 'ATG_HIF1a_CIS_up', 'ATG_HNF4a_TRANS_dn', 'ATG_HNF4a_TRANS_up', 'ATG_HNF6_CIS_dn', 'ATG_HNF6_CIS_up', 'ATG_HSE_CIS_dn', 'ATG_HSE_CIS_up', 'ATG_IR1_CIS_dn', 'ATG_IR1_CIS_up', 'ATG_ISRE_CIS_dn', 'ATG_ISRE_CIS_up', 'ATG_LXRa_TRANS_dn', 'ATG_LXRa_TRANS_up', 'ATG_LXRb_TRANS_dn', 'ATG_LXRb_TRANS_up', 'ATG_MRE_CIS_up', 'ATG_M_06_TRANS_up', 'ATG_M_19_CIS_dn', 'ATG_M_19_TRANS_dn', 'ATG_M_19_TRANS_up', 'ATG_M_32_CIS_dn', 'ATG_M_32_CIS_up', 'ATG_M_32_TRANS_dn', 'ATG_M_32_TRANS_up', 'ATG_M_61_TRANS_up', 'ATG_Myb_CIS_dn', 'ATG_Myb_CIS_up', 'ATG_Myc_CIS_dn', 'ATG_Myc_CIS_up', 'ATG_NFI_CIS_dn', 'ATG_NFI_CIS_up', 'ATG_NF_kB_CIS_dn', 'ATG_NF_kB_CIS_up', 'ATG_NRF1_CIS_dn', 'ATG_NRF1_CIS_up', 'ATG_NRF2_ARE_CIS_dn', 'ATG_NRF2_ARE_CIS_up', 'ATG_NURR1_TRANS_dn', 'ATG_NURR1_TRANS_up', 'ATG_Oct_MLP_CIS_dn', 'ATG_Oct_MLP_CIS_up', 'ATG_PBREM_CIS_dn', 'ATG_PBREM_CIS_up', 'ATG_PPARa_TRANS_dn', 'ATG_PPARa_TRANS_up', 'ATG_PPARd_TRANS_up', 'ATG_PPARg_TRANS_up', 'ATG_PPRE_CIS_dn', 'ATG_PPRE_CIS_up', 'ATG_PXRE_CIS_dn', 'ATG_PXRE_CIS_up', 'ATG_PXR_TRANS_dn', 'ATG_PXR_TRANS_up', 'ATG_Pax6_CIS_up', 'ATG_RARa_TRANS_dn', 'ATG_RARa_TRANS_up', 'ATG_RARb_TRANS_dn', 'ATG_RARb_TRANS_up', 'ATG_RARg_TRANS_dn', 'ATG_RARg_TRANS_up', 'ATG_RORE_CIS_dn', 'ATG_RORE_CIS_up', 'ATG_RORb_TRANS_dn', 'ATG_RORg_TRANS_dn', 'ATG_RORg_TRANS_up', 'ATG_RXRa_TRANS_dn', 'ATG_RXRa_TRANS_up', 'ATG_RXRb_TRANS_dn', 'ATG_RXRb_TRANS_up', 'ATG_SREBP_CIS_dn', 'ATG_SREBP_CIS_up', 'ATG_STAT3_CIS_dn', 'ATG_STAT3_CIS_up', 'ATG_Sox_CIS_dn', 'ATG_Sox_CIS_up', 'ATG_Sp1_CIS_dn', 'ATG_Sp1_CIS_up', 'ATG_TAL_CIS_dn', 'ATG_TAL_CIS_up', 'ATG_TA_CIS_dn', 'ATG_TA_CIS_up', 'ATG_TCF_b_cat_CIS_dn', 'ATG_TCF_b_cat_CIS_up', 'ATG_TGFb_CIS_dn', 'ATG_TGFb_CIS_up', 'ATG_THRa1_TRANS_dn', 'ATG_THRa1_TRANS_up', 'ATG_VDRE_CIS_dn', 'ATG_VDRE_CIS_up', 'ATG_VDR_TRANS_dn', 'ATG_VDR_TRANS_up', 'ATG_XTT_Cytotoxicity_up', 'ATG_Xbp1_CIS_dn', 'ATG_Xbp1_CIS_up', 'ATG_p53_CIS_dn', 'ATG_p53_CIS_up', 'BSK_3C_Eselectin_down', 'BSK_3C_HLADR_down', 'BSK_3C_ICAM1_down', 'BSK_3C_IL8_down', 'BSK_3C_MCP1_down', 'BSK_3C_MIG_down', 'BSK_3C_Proliferation_down', 'BSK_3C_SRB_down', 'BSK_3C_Thrombomodulin_down', 'BSK_3C_Thrombomodulin_up', 'BSK_3C_TissueFactor_down', 'BSK_3C_TissueFactor_up', 'BSK_3C_VCAM1_down', 'BSK_3C_Vis_down', 'BSK_3C_uPAR_down', 'BSK_4H_Eotaxin3_down', 'BSK_4H_MCP1_down', 'BSK_4H_Pselectin_down', 'BSK_4H_Pselectin_up', 'BSK_4H_SRB_down', 'BSK_4H_VCAM1_down', 'BSK_4H_VEGFRII_down', 'BSK_4H_uPAR_down', 'BSK_4H_uPAR_up', 'BSK_BE3C_HLADR_down', 'BSK_BE3C_IL1a_down', 'BSK_BE3C_IP10_down', 'BSK_BE3C_MIG_down', 'BSK_BE3C_MMP1_down', 'BSK_BE3C_MMP1_up', 'BSK_BE3C_PAI1_down', 'BSK_BE3C_SRB_down', 'BSK_BE3C_TGFb1_down', 'BSK_BE3C_tPA_down', 'BSK_BE3C_uPAR_down', 'BSK_BE3C_uPAR_up', 'BSK_BE3C_uPA_down', 'BSK_CASM3C_HLADR_down', 'BSK_CASM3C_IL6_down', 'BSK_CASM3C_IL6_up', 'BSK_CASM3C_IL8_down', 'BSK_CASM3C_LDLR_down', 'BSK_CASM3C_LDLR_up', 'BSK_CASM3C_MCP1_down', 'BSK_CASM3C_MCP1_up', 'BSK_CASM3C_MCSF_down', 'BSK_CASM3C_MCSF_up', 'BSK_CASM3C_MIG_down', 'BSK_CASM3C_Proliferation_down', 'BSK_CASM3C_Proliferation_up', 'BSK_CASM3C_SAA_down', 'BSK_CASM3C_SAA_up', 'BSK_CASM3C_SRB_down', 'BSK_CASM3C_Thrombomodulin_down', 'BSK_CASM3C_Thrombomodulin_up', 'BSK_CASM3C_TissueFactor_down', 'BSK_CASM3C_VCAM1_down', 'BSK_CASM3C_VCAM1_up', 'BSK_CASM3C_uPAR_down', 'BSK_CASM3C_uPAR_up', 'BSK_KF3CT_ICAM1_down', 'BSK_KF3CT_IL1a_down', 'BSK_KF3CT_IP10_down', 'BSK_KF3CT_IP10_up', 'BSK_KF3CT_MCP1_down', 'BSK_KF3CT_MCP1_up', 'BSK_KF3CT_MMP9_down', 'BSK_KF3CT_SRB_down', 'BSK_KF3CT_TGFb1_down', 'BSK_KF3CT_TIMP2_down', 'BSK_KF3CT_uPA_down', 'BSK_LPS_CD40_down', 'BSK_LPS_Eselectin_down', 'BSK_LPS_Eselectin_up', 'BSK_LPS_IL1a_down', 'BSK_LPS_IL1a_up', 'BSK_LPS_IL8_down', 'BSK_LPS_IL8_up', 'BSK_LPS_MCP1_down', 'BSK_LPS_MCSF_down', 'BSK_LPS_PGE2_down', 'BSK_LPS_PGE2_up', 'BSK_LPS_SRB_down', 'BSK_LPS_TNFa_down', 'BSK_LPS_TNFa_up', 'BSK_LPS_TissueFactor_down', 'BSK_LPS_TissueFactor_up', 'BSK_LPS_VCAM1_down', 'BSK_SAg_CD38_down', 'BSK_SAg_CD40_down', 'BSK_SAg_CD69_down', 'BSK_SAg_Eselectin_down', 'BSK_SAg_Eselectin_up', 'BSK_SAg_IL8_down', 'BSK_SAg_IL8_up', 'BSK_SAg_MCP1_down', 'BSK_SAg_MIG_down', 'BSK_SAg_PBMCCytotoxicity_down', 'BSK_SAg_PBMCCytotoxicity_up', 'BSK_SAg_Proliferation_down', 'BSK_SAg_SRB_down', 'BSK_hDFCGF_CollagenIII_down', 'BSK_hDFCGF_EGFR_down', 'BSK_hDFCGF_EGFR_up', 'BSK_hDFCGF_IL8_down', 'BSK_hDFCGF_IP10_down', 'BSK_hDFCGF_MCSF_down', 'BSK_hDFCGF_MIG_down', 'BSK_hDFCGF_MMP1_down', 'BSK_hDFCGF_MMP1_up', 'BSK_hDFCGF_PAI1_down', 'BSK_hDFCGF_Proliferation_down', 'BSK_hDFCGF_SRB_down', 'BSK_hDFCGF_TIMP1_down', 'BSK_hDFCGF_VCAM1_down', 'CEETOX_H295R_11DCORT_dn', 'CEETOX_H295R_ANDR_dn', 'CEETOX_H295R_CORTISOL_dn', 'CEETOX_H295R_DOC_dn', 'CEETOX_H295R_DOC_up', 'CEETOX_H295R_ESTRADIOL_dn', 'CEETOX_H295R_ESTRADIOL_up', 'CEETOX_H295R_ESTRONE_dn', 'CEETOX_H295R_ESTRONE_up', 'CEETOX_H295R_OHPREG_up', 'CEETOX_H295R_OHPROG_dn', 'CEETOX_H295R_OHPROG_up', 'CEETOX_H295R_PROG_up', 'CEETOX_H295R_TESTO_dn', 'CLD_ABCB1_48hr', 'CLD_ABCG2_48hr', 'CLD_CYP1A1_24hr', 'CLD_CYP1A1_48hr', 'CLD_CYP1A1_6hr', 'CLD_CYP1A2_24hr', 'CLD_CYP1A2_48hr', 'CLD_CYP1A2_6hr', 'CLD_CYP2B6_24hr', 'CLD_CYP2B6_48hr', 'CLD_CYP2B6_6hr', 'CLD_CYP3A4_24hr', 'CLD_CYP3A4_48hr', 'CLD_CYP3A4_6hr', 'CLD_GSTA2_48hr', 'CLD_SULT2A_24hr', 'CLD_SULT2A_48hr', 'CLD_UGT1A1_24hr', 'CLD_UGT1A1_48hr', 'NCCT_HEK293T_CellTiterGLO', 'NCCT_QuantiLum_inhib_2_dn', 'NCCT_QuantiLum_inhib_dn', 'NCCT_TPO_AUR_dn', 'NCCT_TPO_GUA_dn', 'NHEERL_ZF_144hpf_TERATOSCORE_up', 'NVS_ADME_hCYP19A1', 'NVS_ADME_hCYP1A1', 'NVS_ADME_hCYP1A2', 'NVS_ADME_hCYP2A6', 'NVS_ADME_hCYP2B6', 'NVS_ADME_hCYP2C19', 'NVS_ADME_hCYP2C9', 'NVS_ADME_hCYP2D6', 'NVS_ADME_hCYP3A4', 'NVS_ADME_hCYP4F12', 'NVS_ADME_rCYP2C12', 'NVS_ENZ_hAChE', 'NVS_ENZ_hAMPKa1', 'NVS_ENZ_hAurA', 'NVS_ENZ_hBACE', 'NVS_ENZ_hCASP5', 'NVS_ENZ_hCK1D', 'NVS_ENZ_hDUSP3', 'NVS_ENZ_hES', 'NVS_ENZ_hElastase', 'NVS_ENZ_hFGFR1', 'NVS_ENZ_hGSK3b', 'NVS_ENZ_hMMP1', 'NVS_ENZ_hMMP13', 'NVS_ENZ_hMMP2', 'NVS_ENZ_hMMP3', 'NVS_ENZ_hMMP7', 'NVS_ENZ_hMMP9', 'NVS_ENZ_hPDE10', 'NVS_ENZ_hPDE4A1', 'NVS_ENZ_hPDE5', 'NVS_ENZ_hPI3Ka', 'NVS_ENZ_hPTEN', 'NVS_ENZ_hPTPN11', 'NVS_ENZ_hPTPN12', 'NVS_ENZ_hPTPN13', 'NVS_ENZ_hPTPN9', 'NVS_ENZ_hPTPRC', 'NVS_ENZ_hSIRT1', 'NVS_ENZ_hSIRT2', 'NVS_ENZ_hTrkA', 'NVS_ENZ_hVEGFR2', 'NVS_ENZ_oCOX1', 'NVS_ENZ_oCOX2', 'NVS_ENZ_rAChE', 'NVS_ENZ_rCNOS', 'NVS_ENZ_rMAOAC', 'NVS_ENZ_rMAOAP', 'NVS_ENZ_rMAOBC', 'NVS_ENZ_rMAOBP', 'NVS_ENZ_rabI2C', 'NVS_GPCR_bAdoR_NonSelective', 'NVS_GPCR_bDR_NonSelective', 'NVS_GPCR_g5HT4', 'NVS_GPCR_gH2', 'NVS_GPCR_gLTB4', 'NVS_GPCR_gLTD4', 'NVS_GPCR_gMPeripheral_NonSelective', 'NVS_GPCR_gOpiateK', 'NVS_GPCR_h5HT2A', 'NVS_GPCR_h5HT5A', 'NVS_GPCR_h5HT6', 'NVS_GPCR_h5HT7', 'NVS_GPCR_hAT1', 'NVS_GPCR_hAdoRA1', 'NVS_GPCR_hAdoRA2a', 'NVS_GPCR_hAdra2A', 'NVS_GPCR_hAdra2C', 'NVS_GPCR_hAdrb1', 'NVS_GPCR_hAdrb2', 'NVS_GPCR_hAdrb3', 'NVS_GPCR_hDRD1', 'NVS_GPCR_hDRD2s', 'NVS_GPCR_hDRD4.4', 'NVS_GPCR_hH1', 'NVS_GPCR_hLTB4_BLT1', 'NVS_GPCR_hM1', 'NVS_GPCR_hM2', 'NVS_GPCR_hM3', 'NVS_GPCR_hM4', 'NVS_GPCR_hNK2', 'NVS_GPCR_hOpiate_D1', 'NVS_GPCR_hOpiate_mu', 'NVS_GPCR_hTXA2', 'NVS_GPCR_p5HT2C', 'NVS_GPCR_r5HT1_NonSelective', 'NVS_GPCR_r5HT_NonSelective', 'NVS_GPCR_rAdra1B', 'NVS_GPCR_rAdra1_NonSelective', 'NVS_GPCR_rAdra2_NonSelective', 'NVS_GPCR_rAdrb_NonSelective', 'NVS_GPCR_rNK1', 'NVS_GPCR_rNK3', 'NVS_GPCR_rOpiate_NonSelective', 'NVS_GPCR_rOpiate_NonSelectiveNa', 'NVS_GPCR_rSST', 'NVS_GPCR_rTRH', 'NVS_GPCR_rV1', 'NVS_GPCR_rabPAF', 'NVS_GPCR_rmAdra2B', 'NVS_IC_hKhERGCh', 'NVS_IC_rCaBTZCHL', 'NVS_IC_rCaDHPRCh_L', 'NVS_IC_rNaCh_site2', 'NVS_LGIC_bGABARa1', 'NVS_LGIC_h5HT3', 'NVS_LGIC_hNNR_NBungSens', 'NVS_LGIC_rGABAR_NonSelective', 'NVS_LGIC_rNNR_BungSens', 'NVS_MP_hPBR', 'NVS_MP_rPBR', 'NVS_NR_bER', 'NVS_NR_bPR', 'NVS_NR_cAR', 'NVS_NR_hAR', 'NVS_NR_hCAR_Antagonist', 'NVS_NR_hER', 'NVS_NR_hFXR_Agonist', 'NVS_NR_hFXR_Antagonist', 'NVS_NR_hGR', 'NVS_NR_hPPARa', 'NVS_NR_hPPARg', 'NVS_NR_hPR', 'NVS_NR_hPXR', 'NVS_NR_hRAR_Antagonist', 'NVS_NR_hRARa_Agonist', 'NVS_NR_hTRa_Antagonist', 'NVS_NR_mERa', 'NVS_NR_rAR', 'NVS_NR_rMR', 'NVS_OR_gSIGMA_NonSelective', 'NVS_TR_gDAT', 'NVS_TR_hAdoT', 'NVS_TR_hDAT', 'NVS_TR_hNET', 'NVS_TR_hSERT', 'NVS_TR_rNET', 'NVS_TR_rSERT', 'NVS_TR_rVMAT2', 'OT_AR_ARELUC_AG_1440', 'OT_AR_ARSRC1_0480', 'OT_AR_ARSRC1_0960', 'OT_ER_ERaERa_0480', 'OT_ER_ERaERa_1440', 'OT_ER_ERaERb_0480', 'OT_ER_ERaERb_1440', 'OT_ER_ERbERb_0480', 'OT_ER_ERbERb_1440', 'OT_ERa_EREGFP_0120', 'OT_ERa_EREGFP_0480', 'OT_FXR_FXRSRC1_0480', 'OT_FXR_FXRSRC1_1440', 'OT_NURR1_NURR1RXRa_0480', 'OT_NURR1_NURR1RXRa_1440', 'TOX21_ARE_BLA_Agonist_ch1', 'TOX21_ARE_BLA_Agonist_ch2', 'TOX21_ARE_BLA_agonist_ratio', 'TOX21_ARE_BLA_agonist_viability', 'TOX21_AR_BLA_Agonist_ch1', 'TOX21_AR_BLA_Agonist_ch2', 'TOX21_AR_BLA_Agonist_ratio', 'TOX21_AR_BLA_Antagonist_ch1', 'TOX21_AR_BLA_Antagonist_ch2', 'TOX21_AR_BLA_Antagonist_ratio', 'TOX21_AR_BLA_Antagonist_viability', 'TOX21_AR_LUC_MDAKB2_Agonist', 'TOX21_AR_LUC_MDAKB2_Antagonist', 'TOX21_AR_LUC_MDAKB2_Antagonist2', 'TOX21_AhR_LUC_Agonist', 'TOX21_Aromatase_Inhibition', 'TOX21_AutoFluor_HEK293_Cell_blue', 'TOX21_AutoFluor_HEK293_Media_blue', 'TOX21_AutoFluor_HEPG2_Cell_blue', 'TOX21_AutoFluor_HEPG2_Cell_green', 'TOX21_AutoFluor_HEPG2_Media_blue', 'TOX21_AutoFluor_HEPG2_Media_green', 'TOX21_ELG1_LUC_Agonist', 'TOX21_ERa_BLA_Agonist_ch1', 'TOX21_ERa_BLA_Agonist_ch2', 'TOX21_ERa_BLA_Agonist_ratio', 'TOX21_ERa_BLA_Antagonist_ch1', 'TOX21_ERa_BLA_Antagonist_ch2', 'TOX21_ERa_BLA_Antagonist_ratio', 'TOX21_ERa_BLA_Antagonist_viability', 'TOX21_ERa_LUC_BG1_Agonist', 'TOX21_ERa_LUC_BG1_Antagonist', 'TOX21_ESRE_BLA_ch1', 'TOX21_ESRE_BLA_ch2', 'TOX21_ESRE_BLA_ratio', 'TOX21_ESRE_BLA_viability', 'TOX21_FXR_BLA_Antagonist_ch1', 'TOX21_FXR_BLA_Antagonist_ch2', 'TOX21_FXR_BLA_agonist_ch2', 'TOX21_FXR_BLA_agonist_ratio', 'TOX21_FXR_BLA_antagonist_ratio', 'TOX21_FXR_BLA_antagonist_viability', 'TOX21_GR_BLA_Agonist_ch1', 'TOX21_GR_BLA_Agonist_ch2', 'TOX21_GR_BLA_Agonist_ratio', 'TOX21_GR_BLA_Antagonist_ch2', 'TOX21_GR_BLA_Antagonist_ratio', 'TOX21_GR_BLA_Antagonist_viability', 'TOX21_HSE_BLA_agonist_ch1', 'TOX21_HSE_BLA_agonist_ch2', 'TOX21_HSE_BLA_agonist_ratio', 'TOX21_HSE_BLA_agonist_viability', 'TOX21_MMP_ratio_down', 'TOX21_MMP_ratio_up', 'TOX21_MMP_viability', 'TOX21_NFkB_BLA_agonist_ch1', 'TOX21_NFkB_BLA_agonist_ch2', 'TOX21_NFkB_BLA_agonist_ratio', 'TOX21_NFkB_BLA_agonist_viability', 'TOX21_PPARd_BLA_Agonist_viability', 'TOX21_PPARd_BLA_Antagonist_ch1', 'TOX21_PPARd_BLA_agonist_ch1', 'TOX21_PPARd_BLA_agonist_ch2', 'TOX21_PPARd_BLA_agonist_ratio', 'TOX21_PPARd_BLA_antagonist_ratio', 'TOX21_PPARd_BLA_antagonist_viability', 'TOX21_PPARg_BLA_Agonist_ch1', 'TOX21_PPARg_BLA_Agonist_ch2', 'TOX21_PPARg_BLA_Agonist_ratio', 'TOX21_PPARg_BLA_Antagonist_ch1', 'TOX21_PPARg_BLA_antagonist_ratio', 'TOX21_PPARg_BLA_antagonist_viability', 'TOX21_TR_LUC_GH3_Agonist', 'TOX21_TR_LUC_GH3_Antagonist', 'TOX21_VDR_BLA_Agonist_viability', 'TOX21_VDR_BLA_Antagonist_ch1', 'TOX21_VDR_BLA_agonist_ch2', 'TOX21_VDR_BLA_agonist_ratio', 'TOX21_VDR_BLA_antagonist_ratio', 'TOX21_VDR_BLA_antagonist_viability', 'TOX21_p53_BLA_p1_ch1', 'TOX21_p53_BLA_p1_ch2', 'TOX21_p53_BLA_p1_ratio', 'TOX21_p53_BLA_p1_viability', 'TOX21_p53_BLA_p2_ch1', 'TOX21_p53_BLA_p2_ch2', 'TOX21_p53_BLA_p2_ratio', 'TOX21_p53_BLA_p2_viability', 'TOX21_p53_BLA_p3_ch1', 'TOX21_p53_BLA_p3_ch2', 'TOX21_p53_BLA_p3_ratio', 'TOX21_p53_BLA_p3_viability', 'TOX21_p53_BLA_p4_ch1', 'TOX21_p53_BLA_p4_ch2', 'TOX21_p53_BLA_p4_ratio', 'TOX21_p53_BLA_p4_viability', 'TOX21_p53_BLA_p5_ch1', 'TOX21_p53_BLA_p5_ch2', 'TOX21_p53_BLA_p5_ratio', 'TOX21_p53_BLA_p5_viability', 'Tanguay_ZF_120hpf_AXIS_up', 'Tanguay_ZF_120hpf_ActivityScore', 'Tanguay_ZF_120hpf_BRAI_up', 'Tanguay_ZF_120hpf_CFIN_up', 'Tanguay_ZF_120hpf_CIRC_up', 'Tanguay_ZF_120hpf_EYE_up', 'Tanguay_ZF_120hpf_JAW_up', 'Tanguay_ZF_120hpf_MORT_up', 'Tanguay_ZF_120hpf_OTIC_up', 'Tanguay_ZF_120hpf_PE_up', 'Tanguay_ZF_120hpf_PFIN_up', 'Tanguay_ZF_120hpf_PIG_up', 'Tanguay_ZF_120hpf_SNOU_up', 'Tanguay_ZF_120hpf_SOMI_up', 'Tanguay_ZF_120hpf_SWIM_up', 'Tanguay_ZF_120hpf_TRUN_up', 'Tanguay_ZF_120hpf_TR_up', 'Tanguay_ZF_120hpf_YSE_up']
    # target_columns: List[str] = ["freesolv"]
    # target_columns: List[str] = ["exp"]
    target_columns: List[str] = ["FDA_APPROVED","CT_TOX"]
    # target_columns: List[str] = ["Hepatobiliary disorders", "Metabolism and nutrition disorders","Product issues","Eye disorders","Investigations","Musculoskeletal and connective tissue disorders","Gastrointestinal disorders","Social circumstances","Immune system disorders","Reproductive system and breast disorders","Neoplasms benign, malignant and unspecified (incl cysts and polyps)","General disorders and administration site conditions","Endocrine disorders","Surgical and medical procedures","Vascular disorders","Blood and lymphatic system disorders","Skin and subcutaneous tissue disorders","Congenital, familial and genetic disorders","Infections and infestations","Respiratory, thoracic and mediastinal disorders","Psychiatric disorders","Renal and urinary disorders","Pregnancy, puerperium and perinatal conditions","Ear and labyrinth disorders","Cardiac disorders","Nervous system disorders","Injury, poisoning and procedural complications"]
    """
    Name of the columns containing target values.
    By default, uses all columns except the SMILES column and the :code:`ignore_columns`.
    """
    message_model: Literal['dmpnn', 'af_dmpnn'] = "af_dmpnn"
    fragment_type: Literal['brics', 'murcko', 'BaseFeatures', 'recap', 'combo'] = "brics"
    ignore_columns: List[str] = None
    """Name of the columns to ignore when :code:`target_columns` is not provided."""
    dataset_type: Literal['regression', 'classification', 'multiclass'] = "classification"
    """Type of dataset. This determines the loss function used during training."""
    multiclass_num_classes: int = 3
    """Number of classes when running multiclass classification."""
    separate_val_path: str = None
    """Path to separate val set, optional."""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    split_type: Literal['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined'] = 'random'
    """Method of splitting the data into train/val/test."""
    split_sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    """Split proportions for train/validation/test sets."""
    num_folds: int = 10
    """Number of folds when performing cross validation."""
    folds_file: str = None
    """Optional file of fold labels."""
    val_fold_index: int = None
    """Which fold to use as val for leave-one-out cross val."""
    test_fold_index: int = None
    """Which fold to use as test for leave-one-out cross val."""
    crossval_index_dir: str = None
    """Directory in which to find cross validation index files."""
    crossval_index_file: str = None
    """Indices of files to use as train/val/test. Overrides :code:`--num_folds` and :code:`--seed`."""
    seed: int = 0
    """
    Random seed to use when splitting data into train/val/test sets.
    When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed.
    """
    pytorch_seed: int = 0
    """Seed for PyTorch randomness (e.g., random initial weights)."""
    metric: Literal['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy'] = None
    """Metric to use during evaluation. Defaults to "auc" for classification and "rmse" for regression."""
    save_smiles_splits: bool = False
    """Save smiles for each train/val/test splits for prediction convenience later."""
    test: bool = False
    """Whether to skip training and only test the model."""
    quiet: bool = False
    """Skip non-essential print statements."""
    log_frequency: int = 10
    """The number of batches between each logging of the training loss."""
    show_individual_scores: bool = False
    """Show all scores for individual targets, not just average, at the end."""
    cache_cutoff: int = 10000
    """
    Maximum number of molecules in dataset to allow caching.
    Below this number, caching is used and data loading is sequential.
    Above this number, caching is not used and data loading is parallel.
    """

    # Model arguments
    bias: bool = False
    """Whether to add bias to linear layers."""
    hidden_size: int = 300
    """Dimensionality of hidden layers in MPN."""
    depth: int = 3
    """Number of message passing steps."""
    dropout: float = 0.1
    """Dropout probability."""
    activation: Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'] = 'ReLU'
    """Activation function."""
    atom_messages: bool = False
    """Centers messages on atoms instead of on bonds."""
    undirected: bool = False
    """Undirected edges (always sum the two relevant bond vectors)."""
    ffn_hidden_size: int = None
    """Hidden dim for higher-capacity FFN (defaults to hidden_size)."""
    ffn_num_layers: int = 2
    """Number of layers in FFN after MPN encoding."""
    features_only: bool = False
    """Use only the additional features in an FFN, no graph network."""
    separate_val_features_path: List[str] = None
    """Path to file with features for separate val set."""
    separate_test_features_path: List[str] = None
    """Path to file with features for separate test set."""
    config_path: str = None
    """
    Path to a :code:`.json` file containing arguments. Any arguments present in the config file
    will override arguments specified via the command line or by the defaults.
    """
    ensemble_size: int = 1
    """Number of models in ensemble."""

    # Training arguments
    epochs: int = 50
    """Number of epochs to run."""
    warmup_epochs: float = 2.0
    """
    Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
    Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
    """
    init_lr: float = 1e-4
    """Initial learning rate."""
    max_lr: float = 1e-3
    """Maximum learning rate."""
    final_lr: float = 1e-4
    """Final learning rate."""
    grad_clip: float = None
    """Maximum magnitude of gradient during training."""
    class_balance: bool = False
    """Trains with an equal number of positives and negatives in each batch (only for single task classification)."""

    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self._task_names = None
        self._crossval_index_sets = None
        self._task_names = None
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None

    @property
    def minimize_score(self) -> bool:
        """Whether the model should try to minimize the score metric or maximize it."""
        return self.metric in {'rmse', 'mae', 'mse', 'cross_entropy'}

    @property
    def use_input_features(self) -> bool:
        """Whether the model is using additional molecule-level features."""

        return (self.features_generator is not None and len(self.features_generator)) or self.features_path is not None

    @property
    def num_lrs(self) -> int:
        """The number of learning rates to use (currently hard-coded to 1)."""
        return 1

    @property
    def crossval_index_sets(self) -> List[List[List[int]]]:
        """Index sets used for splitting data into train/validation/test during cross-validation"""
        return self._crossval_index_sets

    @property
    def task_names(self) -> List[str]:
        """A list of names of the tasks being trained on."""
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        """The number of tasks being trained on."""
        return len(self.task_names) if self.task_names is not None else 0

    @property
    def features_size(self) -> int:
        """The dimensionality of the additional molecule-level features."""
        return self._features_size

    @features_size.setter
    def features_size(self, features_size: int) -> None:
        self._features_size = features_size

    @property
    def train_data_size(self) -> int:
        """The size of the training data set."""
        return self._train_data_size

    @train_data_size.setter
    def train_data_size(self, train_data_size: int) -> None:
        self._train_data_size = train_data_size

    def process_args(self) -> None:
        super(TrainArgs, self).process_args()

        global temp_dir  # Prevents the temporary directory from being deleted upon function return

        # Load config file
        if self.config_path is not None:
            with open(self.config_path) as f:
                config = json.load(f)
                for key, value in config.items():
                    setattr(self, key, value)

        # Create temporary directory as save directory if not provided
        if self.save_dir is None:
            temp_dir = TemporaryDirectory()
            self.save_dir = temp_dir.name

        # Fix ensemble size if loading checkpoints
        if self.checkpoint_paths is not None and len(self.checkpoint_paths) > 0:
            self.ensemble_size = len(self.checkpoint_paths)

        # Process and validate metric and loss function
        if self.metric is None:
            if self.dataset_type == 'classification':
                self.metric = 'auc'
            elif self.dataset_type == 'multiclass':
                self.metric = 'cross_entropy'
            else:
                self.metric = 'rmse'

        if not ((self.dataset_type == 'classification' and self.metric in ['auc', 'prc-auc', 'accuracy']) or
                (self.dataset_type == 'regression' and self.metric in ['rmse', 'mae', 'mse', 'r2']) or
                (self.dataset_type == 'multiclass' and self.metric in ['cross_entropy', 'accuracy'])):
            raise ValueError(f'Metric "{self.metric}" invalid for dataset type "{self.dataset_type}".')

        # Validate class balance
        if self.class_balance and self.dataset_type != 'classification':
            raise ValueError('Class balance can only be applied if the dataset type is classification.')

        # Validate features
        if self.features_only and not (self.features_generator or self.features_path):
            raise ValueError('When using features_only, a features_generator or features_path must be provided.')

        # Handle FFN hidden size
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size

        # Handle MPN variants
        if self.atom_messages and self.undirected:
            raise ValueError('Undirected is unnecessary when using atom_messages '
                             'since atom_messages are by their nature undirected.')

        # Validate split type settings
        if not (self.split_type == 'predetermined') == (self.folds_file is not None) == (self.test_fold_index is not None):
            raise ValueError('When using predetermined split type, must provide folds_file and test_fold_index.')

        if not (self.split_type == 'crossval') == (self.crossval_index_dir is not None):
            raise ValueError('When using crossval split type, must provide crossval_index_dir.')

        if not (self.split_type in ['crossval', 'index_predetermined']) == (self.crossval_index_file is not None):
            raise ValueError('When using crossval or index_predetermined split type, must provide crossval_index_file.')

        if self.split_type in ['crossval', 'index_predetermined']:
            with open(self.crossval_index_file, 'rb') as rf:
                self._crossval_index_sets = pickle.load(rf)
            self.num_folds = len(self.crossval_index_sets)
            self.seed = 0

        # Test settings
        if self.test:
            self.epochs = 0


class PredictArgs(CommonArgs):
    """:class:`PredictArgs` includes :class:`CommonArgs` along with additional arguments used for predicting with a Chemprop model."""

    test_path: str
    """Path to CSV file containing testing data for which predictions will be made."""
    preds_path: str
    """Path to CSV file where predictions will be saved."""

    @property
    def ensemble_size(self) -> int:
        """The number of models in the ensemble."""
        return len(self.checkpoint_paths)

    def process_args(self) -> None:
        super(PredictArgs, self).process_args()

        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError('Found no checkpoints. Must specify --checkpoint_path <path> or '
                             '--checkpoint_dir <dir> containing at least one checkpoint.')


class InterpretArgs(CommonArgs):
    """:class:`InterpretArgs` includes :class:`CommonArgs` along with additional arguments used for interpreting a trained Chemprop model."""

    data_path: str
    """Path to data CSV file."""
    batch_size: int = 500
    """Batch size."""
    property_id: int = 1
    """Index of the property of interest in the trained model."""
    rollout: int = 20
    """Number of rollout steps."""
    c_puct: float = 10.0
    """Constant factor in MCTS."""
    max_atoms: int = 20
    """Maximum number of atoms in rationale."""
    min_atoms: int = 8
    """Minimum number of atoms in rationale."""
    prop_delta: float = 0.5
    """Minimum score to count as positive."""

    def process_args(self) -> None:
        super(InterpretArgs, self).process_args()

        if self.features_path is not None:
            raise ValueError('Cannot use --features_path <path> for interpretation since features '
                             'need to be computed dynamically for molecular substructures. '
                             'Please specify --features_generator <generator>.')

        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError('Found no checkpoints. Must specify --checkpoint_path <path> or '
                             '--checkpoint_dir <dir> containing at least one checkpoint.')


class HyperoptArgs(TrainArgs):
    """:class:`HyperoptArgs` includes :class:`TrainArgs` along with additional arguments used for optimizing Chemprop hyperparameters."""

    num_iters: int = 20
    """Number of hyperparameter choices to try."""
    config_save_path: str = r"model\AF-DMPNN-train/freesolv-t2"
    """Path to :code:`.json` file where best hyperparameter settings will be written."""
    log_dir: str = None
    """(Optional) Path to a directory where all results of the hyperparameter optimization will be written."""


class SklearnTrainArgs(TrainArgs):
    """:class:`SklearnTrainArgs` includes :class:`TrainArgs` along with additional arguments for training a scikit-learn model."""

    model_type: Literal['random_forest', 'svm']
    """scikit-learn model to use."""
    class_weight: Literal['balanced'] = None
    """How to weight classes (None means no class balance)."""
    single_task: bool = False
    """Whether to run each task separately (needed when dataset has null entries)."""
    radius: int = 2
    """Morgan fingerprint radius."""
    num_bits: int = 2048
    """Number of bits in morgan fingerprint."""
    num_trees: int = 500
    """Number of random forest trees."""


class SklearnPredictArgs(Tap):
    """:class:`SklearnPredictArgs` contains arguments used for predicting with a trained scikit-learn model."""

    test_path: str
    """Path to CSV file containing testing data for which predictions will be made."""
    smiles_column: str = None
    """Name of the column containing SMILES strings. By default, uses the first column."""
    preds_path: str
    """Path to CSV file where predictions will be saved."""
    checkpoint_dir: str = None
    """Path to directory containing model checkpoints (:code:`.pkl` file)"""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pkl` file)"""
    checkpoint_paths: List[str] = None
    """List of paths to model checkpoints (:code:`.pkl` files)"""

    def process_args(self) -> None:
        # Load checkpoint paths
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path=self.checkpoint_path,
            checkpoint_paths=self.checkpoint_paths,
            checkpoint_dir=self.checkpoint_dir,
            ext='.pkl'
        )
