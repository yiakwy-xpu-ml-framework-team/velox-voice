"""Character/BPE unit table (units.txt) used for CTC id<->text mapping."""

from __future__ import annotations


class TextTokenizer:
    """Maps CTC ids to symbols using a WeNet-style `units.txt` (one unit per line,
    line index == id; id 0 is <blank>). Space is decoded from the `<space>` unit."""

    def __init__(self, units_path: str):
        self.units: list[str] = []
        with open(units_path, encoding="utf-8") as f:
            for line in f:
                sym = line.strip().split(" ")[0]
                self.units.append(sym)
        self.blank = 0

    @property
    def vocab_size(self) -> int:
        return len(self.units)

    # TODO (yiakwy) : move to GPU
    def ids_to_text(self, ids: list[int]) -> str:
        chars = []
        for i in ids:
            if i == self.blank or i >= len(self.units):
                continue
            sym = self.units[i]
            if sym == "<space>":
                chars.append(" ")
            else:
                # SentencePiece-style space marker used by BPE units ("▁GOD" ...)
                chars.append(sym.replace("▁", " "))
        return "".join(chars)
