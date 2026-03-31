-- Imaginary parts γ_n of nontrivial zeros ρ_n = 1/2 + iγ_n of ζ(s).
CREATE TABLE IF NOT EXISTS zeta_zeros (
    n INTEGER PRIMARY KEY CHECK (n >= 1),
    gamma REAL NOT NULL CHECK (gamma > 0)
);
CREATE INDEX IF NOT EXISTS idx_zeta_zeros_gamma ON zeta_zeros (gamma);
