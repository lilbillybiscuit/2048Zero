#!/usr/bin/env python3

from bitwrapper import BitBoardEnv          
import math, random
from functools import lru_cache

ROW_MASK  = 0xFFFF
FULL_MASK = 0xFFFFFFFFFFFFFFFF

def unpack_row(row16: int):
    """16‑bit row → [e0,e1,e2,e3] exponents (0 = empty)."""
    return [(row16 >> (4 * i)) & 0xF for i in range(4)]

def pack_row(vals):
    """[e0..e3] → 16‑bit row."""
    r = 0
    for i, v in enumerate(vals):
        r |= (v & 0xF) << (4 * i)
    return r

def reflect_row(row16: int) -> int:
    """abcd (LSB→MSB)  ⇒  dcba."""
    return ((row16 & 0xF)      << 12 |
            (row16 & 0xF0)     << 4  |
            (row16 & 0xF00)    >> 4  |
            (row16 & 0xF000)   >> 12)

def transpose(board: int) -> int:
    """Swap rows ↔ columns (Hacker’s Delight §7‑1)."""
    a1 = board & 0xF0F00F0FF0F00F0F
    a2 = board & 0x0000F0F00000F0F0
    a3 = board & 0x0F0F00000F0F0000
    a2 = (a2 << 12) | (a2 >> 20)
    a3 = (a3 << 20) | (a3 >> 12)
    return a1 | a2 | a3

def reflect(board: int) -> int:
    """Mirror every row horizontally."""
    r = 0
    for i in range(4):
        row = (board >> (16 * i)) & ROW_MASK
        r  |= reflect_row(row) << (16 * i)
    return r

def count_empty(board: int) -> int:
    """Fast popcount of zero nibbles."""
    x = board | ((board >> 1) & 0x7777777777777777)
    x |= (x >> 2)
    empties_mask = ~x & 0x1111111111111111
    return (empties_mask * 0x1111111111111111) >> 60   # 0‑16

ROW_MOVE  = [0] * 65536
ROW_SCORE = [0] * 65536
PART_MRG  = [0] * 65536
PART_SMO  = [0] * 65536
PART_MONO = [0] * 65536     

def _init_tables():
    for r in range(65536):
        cells = unpack_row(r)
        score = 0

        tight = [c for c in cells if c]
        i = 0
        while i < len(tight) - 1:
            if tight[i] == tight[i + 1]:
                tight[i] += 1
                score += 1 << tight[i]         
                tight.pop(i + 1)
            i += 1
        tight += [0] * (4 - len(tight))
        ROW_MOVE[r]  = pack_row(tight)
        ROW_SCORE[r] = score

        PART_MRG[r] = sum(
            1 for i in range(3) if cells[i] and cells[i] == cells[i + 1]
        )
        PART_SMO[r] = sum(
            abs(cells[i] - cells[i + 1])
            for i in range(3)
            if cells[i] and cells[i + 1]
        )
        inc = sum((cells[i] - cells[i + 1]) for i in range(3) if cells[i] > cells[i + 1])
        dec = sum((cells[i + 1] - cells[i]) for i in range(3) if cells[i] < cells[i + 1])
        PART_MONO[r] = max(inc, dec)

_init_tables()

def row_left(board: int):
    """Return (new_board, score_gain) for a left move on all rows."""
    res   = board
    gain  = 0
    for i in range(4):
        row = (board >> (16 * i)) & ROW_MASK
        merged = ROW_MOVE[row]
        res    &= ~(ROW_MASK << (16 * i))
        res    |= merged << (16 * i)
        gain   += ROW_SCORE[row]
    return res, gain

def row_right(board: int):
    t, g = row_left(reflect(board))
    return reflect(t), g

def move(board: int, dir_: int):
    """dir_: 0=Up 1=Right 2=Down 3=Left. Return (new_board, gain)."""
    if dir_ == 3:                      
        nb, sc = row_left(board)
    elif dir_ == 1:                   
        nb, sc = row_right(board)
    else:                             
        tb = transpose(board)
        if dir_ == 0:                  
            nb, sc = row_left(tb)
        else:                         
            nb, sc = row_right(tb)
        nb = transpose(nb)
    return (nb, sc) if nb != board else (board, 0)


def heuristic(board: int, α=-0.1, β=1.0, γ=2.7, δ=3.5):
    """Lower is worse; expectimax maximises −heuristic."""
    empt  = count_empty(board)
    smooth = mono = merges = 0
    # import ipdb; ipdb.set_trace()

    for i in range(4):
        row = (board >> (16 * i)) & ROW_MASK
        smooth += PART_SMO[row]
        mono   += PART_MONO[row]
        merges += PART_MRG[row]

    t = transpose(board)
    for i in range(4):
        row = (t >> (16 * i)) & ROW_MASK
        smooth += PART_SMO[row]
        mono   += PART_MONO[row]
        merges += PART_MRG[row]

    return α * smooth + β * mono + γ * empt + δ * merges

@lru_cache(maxsize=None)
def _spawn_children(b: int):
    """Return [(child_board, prob), ...] after random spawn."""
    empt = [(b >> (4 * k)) & 0xF == 0 for k in range(16)]
    spots = [k for k, e in enumerate(empt) if e]
    if not spots:
        return []
    res = []
    for k in spots:
        cellmask = 0xF << (4 * k)
        for val, p in ((1, 0.9), (2, 0.1)):    
            res.append(((b & ~cellmask) | (val << (4 * k)), p / len(spots)))
    return res

def expectimax(board: int, depth: int):
    """Return (best_dir, expected_value)."""
    def max_node(b, d):
        best = (-1, -math.inf)
        for dir_ in range(4):
            nb, sc = move(b, dir_)
            if nb == b:
                continue
            val = sc + (heuristic(nb) if d == 0 else chance_node(nb, d - 1))
            if val > best[1]:
                best = (dir_, val)
        return best

    def chance_node(b, d):
        exp = 0.0
        for cb, p in _spawn_children(b):
            exp += p * (heuristic(cb) if d == 0 else max_node(cb, d - 1)[1])
        return exp

    return max_node(board, depth)

def play(depth: int = 3, verbose: bool = True, seed: int | None = None) -> int:
    if seed is not None:
        random.seed(seed)
        
    env = BitBoardEnv()
    b   = env.b
    score = 0

    while True:
        direction, _ = expectimax(b, depth)
        if direction == -1:
            break

        new_b, gained = env.move(direction)
        if new_b == b:
            break               

        score += gained
        b = new_b

        if env.count_empty() == 0:
            break
        env.add_random_tile()
        b = env.b

        print(f"\nMove {['↑','→','↓','←'][direction]}  (+{gained})  total={score}")
        for r in range(4):
            cells = []
            for c in range(4):
                e = (b >> (4 * (r * 4 + c))) & 0xF
                cells.append(".".rjust(5) if e == 0 else str(1 << e).rjust(5))
            print(" ".join(cells))

    # print(f"\nGame over – final score: {score}")
    return score


if __name__ == "__main__":
    play(depth=3)
