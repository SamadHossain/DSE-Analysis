"""Paths and shared constants.

Paths are resolved relative to the repository root so the code runs the same
way locally, in Docker, or in a hosted notebook.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

RAW_DATA_FILE = RAW_DATA_DIR / "DSE_Data.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_stock_data.csv"

RANDOM_STATE = 42

# Chronological split boundaries. Everything up to TRAIN_END is used for
# fitting, the following year for early stopping, the remainder for testing.
TRAIN_END = "2022-12-31"
VALID_END = "2023-12-31"

# Non-equity instruments excluded from the analysis: the three DSE indices,
# corporate/perpetual bonds, and closed-end mutual funds. Their price series
# behave differently from ordinary shares, so mixing them into a single
# cross-sectional model is not appropriate.
NON_EQUITY_CODES = [
    "DSES", "DS30", "DSEX",
    "BEXGSUKUK", "IBBLPBOND", "AIBLPBOND", "PBLPBOND", "SJIBLPBOND",
    "IBBL2PBOND", "APSCLBOND", "ABBLPBOND", "SEB1PBOND", "DBLPBOND",
    "PREBPBOND", "MBPLCPBOND", "BANKASI1PB", "UCB2PBOND", "CBLPBOND",
    "BRACSCBOND", "ACIZCBOND",
    "FBFIF", "GLDNJMF", "ICBSONALI1", "ICBEPMF1S1", "ABB1STMF", "PF1STMF",
    "POPULAR1MF", "RELIANCE1", "EBL1STMF", "LRGLOBMF1", "EBLNRBMF",
    "TRUSTB1MF", "ICB3RDNRB", "MBL1STMF", "GRAMEENS2", "GREENDELMF",
    "ICBAMCL2ND", "NCCBLMF1", "AIBL1STIMF", "SEMLFBSLGF", "SEMLLECMF",
    "DBH1STMF", "SEMLIBBLSF", "PHPMF1", "ICBAGRANI1", "CAPITECGBF",
    "PRIME1ICBA", "EXIM1STMF", "1JANATAMF", "CAPMIBBLMF", "VAMLRBBF",
    "IFIC1STMF", "IFILISLMF1", "CAPMBDBLMF", "VAMLBDMF1", "ATCSLGF",
    "NLI1STMF", "SEBL1STMF", "ICB2NDNRB", "ICB1STNRB", "GRAMEEN1",
    "AIMS1STMF", "ICBAMCL1ST",
]

PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
OHLCV_COLUMNS = PRICE_COLUMNS + ["Volume"]
