import random

# =====================================================================
# CUSTOM KITTY SELECTION
# =====================================================================
# Set this to the kitty you want to practice on!
#   "chess"     -> Iracebeth (Knight/Bishop moves)
#   "edge"      -> Peter (Perimeter/Wallhugger)
#   "skittish" -> Jigu (Flees when approached)
#   "matador"   -> Manolo (Center arena & dodge)
#   "random"    -> Pick a random kitty per run
# =====================================================================
SELECTED_KITTY = "matador"  # <--- CHANGE THIS TO YOUR CHOICE!
# =====================================================================


def move_chess(cat):
    """
        The cat is programmed to use chess moves like a Knight (L-shape) or Bishop (Diagonal).
    """
    knight_moves = [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]
    bishop_moves = [(1,1), (1,-1), (-1,1), (-1,-1)]
    all_moves = knight_moves + bishop_moves
    random.shuffle(all_moves)
    
    for dr, dc in all_moves:
        new_r = cat.pos[0] + dr
        new_c = cat.pos[1] + dc
        if 0 <= new_r < cat.grid_size and 0 <= new_c < cat.grid_size:
            cat.pos[0] = new_r
            cat.pos[1] = new_c
            break

def move_wallhugger(cat):
    """
        The cat is programmed to stick to the nearest edge and slides along the perimeter (aka it just moves along the walls)
    """
    if cat.pos[0] not in (0, cat.grid_size - 1) and cat.pos[1] not in (0, cat.grid_size - 1):
        dist_to_top, dist_to_bottom = cat.pos[0], (cat.grid_size - 1) - cat.pos[0]
        dist_to_left, dist_to_right = cat.pos[1], (cat.grid_size - 1) - cat.pos[1]
        
        min_dist = min(dist_to_top, dist_to_bottom, dist_to_left, dist_to_right)
        
        if min_dist == dist_to_top: cat.pos[0] -= 1
        elif min_dist == dist_to_bottom: cat.pos[0] += 1
        elif min_dist == dist_to_left: cat.pos[1] -= 1
        else: cat.pos[1] += 1
    else:
        moves = []
        if cat.pos[0] == 0 or cat.pos[0] == cat.grid_size - 1:
            if cat.pos[1] > 0: moves.append((0, -1))
            if cat.pos[1] < cat.grid_size - 1: moves.append((0, 1))
        if cat.pos[1] == 0 or cat.pos[1] == cat.grid_size - 1:
            if cat.pos[0] > 0: moves.append((-1, 0))
            if cat.pos[0] < cat.grid_size - 1: moves.append((1, 0))
            
        if moves:
            dr, dc = random.choice(moves)
            cat.pos[0] += dr
            cat.pos[1] += dc

def move_skittish(cat):
    """
        This cat flees when approached. If CatBot is within 3 tiles, it will flee to the farthest valid tile. 
        Otherwise, it has a 30% chance to move around randomly.
    """
    r, c = cat.pos
    pr, pc = cat.player_pos
    dist = cat.current_distance
    
    candidates = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < cat.grid_size and 0 <= nc < cat.grid_size:
            candidates.append((nr, nc))
            
    if not candidates:
        return

    if dist <= 3:
        best_moves = []
        max_dist = -1
        
        for nr, nc in candidates:
            new_dist = abs(nr - pr) + abs(nc - pc)
            if new_dist > max_dist:
                max_dist = new_dist
                best_moves = [(nr, nc)]
            elif new_dist == max_dist:
                best_moves.append((nr, nc))
                
        if best_moves:
            chosen = random.choice(best_moves)
            cat.pos[0] = chosen[0]
            cat.pos[1] = chosen[1]
            return

    elif random.random() < 0.3:
        chosen = random.choice(candidates)
        cat.pos[0] = chosen[0]
        cat.pos[1] = chosen[1]

def move_matador(cat):
    """
        This cat is a true Sanchez bullfighter. It stays at the center of the arena 
        and attempts to dodge CatBot's moves. If CatBot is within 2 tiles, it will try to sidestep perpendicularly.
        Otherwise, it has a 60% chance to casually pace toward the center of the arena
    """
    r, c = cat.pos
    pr, pc = cat.player_pos
    dist = cat.current_distance
    
    candidates = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < cat.grid_size and 0 <= nc < cat.grid_size:
            candidates.append((dr, dc, nr, nc))
            
    if not candidates:
        return

    if dist <= 2:
        # 75% chance to succeed in a perfect perpendicular dodge
        if random.random() < 0.75:
            row_diff = pr - r
            col_diff = pc - c
            prefer_horizontal = abs(row_diff) >= abs(col_diff)
            
            valid_dodges = []
            for dr, dc, nr, nc in candidates:
                new_dist = abs(nr - pr) + abs(nc - pc)
                if new_dist >= dist:
                    is_horizontal = (dc != 0)
                    score = 2 if (is_horizontal == prefer_horizontal) else 1
                    valid_dodges.append((score, new_dist, nr, nc))
                    
            if valid_dodges:
                valid_dodges.sort(key=lambda x: (x[0], x[1]), reverse=True)
                top_score = valid_dodges[0][0]
                best_dodges = [d for d in valid_dodges if d[0] == top_score]
                
                chosen = random.choice(best_dodges)
                cat.pos[0] = chosen[2]
                cat.pos[1] = chosen[3]
                return
                
        # If the dodge fails (25% fumble), do a basic flee away from CatBot
        best_moves = []
        max_dist = -1
        for dr, dc, nr, nc in candidates:
            d = abs(nr - pr) + abs(nc - pc)
            if d > max_dist:
                max_dist = d
                best_moves = [(nr, nc)]
            elif d == max_dist:
                best_moves.append((nr, nc))
                
        if best_moves:
            chosen = random.choice(best_moves)
            cat.pos[0] = chosen[0]
            cat.pos[1] = chosen[1]
            return

    center_r = cat.grid_size // 2
    center_c = cat.grid_size // 2
    
    # 60% chance to casually pace toward the center if he isn't already there
    if (r != center_r or c != center_c) and random.random() < 0.60:
        best_moves = []
        min_center_dist = 999
        
        for dr, dc, nr, nc in candidates:
            cdist = abs(nr - center_r) + abs(nc - center_c)
            if cdist < min_center_dist:
                min_center_dist = cdist
                best_moves = [(nr, nc)]
            elif cdist == min_center_dist:
                best_moves.append((nr, nc))
        
        if best_moves:
            chosen = random.choice(best_moves)
            cat.pos[0] = chosen[0]
            cat.pos[1] = chosen[1]

BEHAVIOR_DICT = {
    "chess": ("Iracebeth (Chess Cat)", move_chess),
    "edge": ("Peter (Edge Cat)", move_wallhugger),
    "skittish": ("Jigu (Skittish Cat)", move_skittish),
    "matador": ("Manolo (Matador Cat)", move_matador)
}

def get_locked_behavior():
    """Returns the chosen behavior locked for this script execution."""
    key = SELECTED_KITTY.lower().strip()
    if key in BEHAVIOR_DICT:
        return BEHAVIOR_DICT[key]
    else:
        # fallback to random if "random" or an unknown option is selected
        name, func = random.choice(list(BEHAVIOR_DICT.values()))
        return f"{name} [Random Selection]", func