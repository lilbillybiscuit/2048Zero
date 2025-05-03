#!/usr/bin/env python3
# 2048_bitboard.py

import random
import math
from functools import lru_cache

ROW_MASK     = 0xFFFF                     # 16 bits
COL_MASK     = 0x000F000F000F000F         # column 0
FULL_MASK    = 0xFFFFFFFFFFFFFFFF

def unpack_row(row16):
    """Return list[4] of cell exponents (0 = empty, 1 → 2, 2 → 4 …)."""
    return [(row16 >> (4 * i)) & 0xF for i in range(4)]

def pack_row(vals):
    """vals iterable of 4 ints (0‑15) → 16‑bit row."""
    r = 0
    for i, v in enumerate(vals):
        r |= (v & 0xF) << (4 * i)
    return r

def reflect_row(row16):
    """abcd (from LSB) → dcba (bit‑wise mirror)."""
    return ((row16 & 0xF)        << 12 |
            (row16 & 0xF0)       << 4  |
            (row16 & 0xF00)      >> 4  |
            (row16 & 0xF000)     >> 12)

def transpose(board):
    """Swap rows↔cols using bit tricks (Hacker’s Delight §7‑1)."""
    a1 = board & 0xF0F00F0FF0F00F0F
    a2 = board & 0x0000F0F00000F0F0
    a3 = board & 0x0F0F00000F0F0000
    a2 = (a2 << 12) | (a2 >> 20)
    a3 = (a3 << 20) | (a3 >> 12)
    return a1 | a2 | a3

def count_empty(board):
    """Number of zero cells (fast popcount on complemented 4‑bit nibbles)."""
    
    x = board | ((board >> 1) & 0x7777777777777777)
    x |= (x >> 2)
    empties_mask = ~x & 0x1111111111111111
    
    return (empties_mask * 0x1111111111111111) >> 60


ROW_MOVE  = [0]  * 65536         
ROW_SCORE = [0]  * 65536         
PART_MRG  = [0]  * 65536         
PART_SMO  = [0]  * 65536         
PART_MONO = [0]  * 65536         

def _init_tables():
    for r in range(65536):
        cells = unpack_row(r)
        score = 0

        tight = [c for c in cells if c]
        for i in range(len(tight)-1):
            if tight[i] == tight[i+1]:
                tight[i] += 1
                score += (1 << tight[i])  
                tight[i+1] = 0
        tight = [c for c in tight if c] + [0]*(4-len(tight))
        ROW_MOVE[r]  = pack_row(tight)
        ROW_SCORE[r] = score

        merge_pot = sum(1 for i in range(3) if cells[i] and cells[i]==cells[i+1])
        PART_MRG[r] = merge_pot

        smooth = sum(abs(cells[i]-cells[i+1]) for i in range(3)
                     if cells[i] and cells[i+1])
        PART_SMO[r] = smooth

        mono_inc = sum((cells[i] - cells[i+1])
                       for i in range(3) if cells[i] > cells[i+1])
        mono_dec = sum((cells[i+1] - cells[i])
                       for i in range(3) if cells[i] < cells[i+1])
        PART_MONO[r] = max(mono_inc, mono_dec)

_init_tables()

def row_left(board):
    """Return (new_board, score_delta)."""
    res = board
    score = 0
    for i in range(4):
        import ipdb; ipdb.set_trace()
        row = (board >> (16*i)) & ROW_MASK
        merged = ROW_MOVE[row]
        res  &= ~(ROW_MASK << (16*i))
        res  |= merged << (16*i)
        score += ROW_SCORE[row]
    return res, score

def row_right(board):
    t, s = row_left(reflect(board))
    return reflect(t), s

def reflect(board):
    
    r = 0
    for i in range(4):
        row = (board >> (16*i)) & ROW_MASK
        r |= reflect_row(row) << (16*i)
    return r

def move(board, dir):
    """
    dir ∈ {0:Up,1:Right,2:Down,3:Left}
    Returns new_board, score_delta. If move is illegal → (board,0)
    """
    if dir == 3:     
        new, sc = row_left(board)
    elif dir == 1:  
        new, sc = row_right(board)
    else:
        b = transpose(board)
        if dir == 0:        
            new, sc = row_left(b)
        else:              
            new, sc = row_right(b)
        new = transpose(new)
    return (new, sc) if new != board else (board, 0)


def heuristic(board, α=-0.1, β=1.0, γ=2.7, δ=3.5):
    empt = count_empty(board)
    smooth = mono = merges = 0

    for i in range(4):
        row = (board >> (16*i)) & ROW_MASK
        smooth += PART_SMO[row]
        mono   += PART_MONO[row]
        merges += PART_MRG[row]


    t = transpose(board)
    for i in range(4):
        row = (t >> (16*i)) & ROW_MASK
        smooth += PART_SMO[row]
        mono   += PART_MONO[row]
        merges += PART_MRG[row]

    return α*smooth + β*mono + γ*empt + δ*merges


@lru_cache(maxsize=None)
def _spawn_children(b):
    """Return list[(board,prob)] after adding 2‑tile (p=0.9) or 4‑tile (p=0.1)."""
    empties = [(b >> (4*k)) & 0xF == 0 for k in range(16)]
    spots   = [k for k,e in enumerate(empties) if e]
    res = []
    for k in spots:
        cellmask = 0xF << (4*k)
        for val, p in ((1, 0.9), (2, 0.1)):       # 2 or 4
            res.append(((b & ~cellmask) | (val << (4*k)), p/len(spots)))
    return res

def expectimax(board, depth=3):
    """Return (best_dir,value)."""
    def _max_node(b, d):
        best = (-1, -math.inf)
        for dir in range(4):
            nb, sc = move(b, dir)
            if sc == 0 and nb == b:
                continue
            v = sc + (heuristic(nb) if d==0 else _chance_node(nb, d-1))
            if v > best[1]:
                best = (dir, v)
        return best

    def _chance_node(b, d):
        exp = 0.0
        for cb, p in _spawn_children(b):
            if d == 0:
                exp += p * heuristic(cb)
            else:
                exp += p * _max_node(cb, d-1)[1]
        return exp

    return _max_node(board, depth)

def as_grid(b):
    return [[(1 << ((b >> (4*(i*4+j))) & 0xF)) if ((b >> (4*(i*4+j))) & 0xF) else 0
             for j in range(4)] for i in range(4)]

def random_tile(b):
    empties = [k for k in range(16) if ((b >> (4*k)) & 0xF) == 0]
    k = random.choice(empties)
    v = 1 if random.random() < 0.9 else 2
    return b | (v << (4*k))

def play():
    b = random_tile(random_tile(0))
    score = 0
    while True:
        d, _ = expectimax(b, depth=3)
        if d == -1:                       
            break

        new_b, gained = move(b, d)
        if new_b == b:                   
            break

        b, score = new_b, score + gained
        import ipdb; ipdb.set_trace()

        if count_empty(b) == 0:           
            break
        b = random_tile(b)                

        print(f"\nMove {['↑','→','↓','←'][d]}, score+{gained}  total={score}")
        for row in as_grid(b):
            print(" ".join(f"{x:5d}" for x in row))

    print(f"\nGame over. Final score: {score}")


if __name__ == "__main__":
    play()
