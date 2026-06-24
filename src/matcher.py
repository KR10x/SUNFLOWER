from collections import defaultdict

def find_match(q_hashes, db_obj):
    tally = defaultdict(lambda: defaultdict(int))
    
    for h_tag, q_t in q_hashes:
        if h_tag in db_obj.stored_hashes:
            found = db_obj.stored_hashes[h_tag]
            for title, db_t in found:
                t_diff = db_t - q_t
                tally[title][t_diff] += 1
                
    top_track = None
    max_hits = 0
    best_hist = None
    
    for title, offset_dict in tally.items():
        if not offset_dict:
            continue
            
        best_diff = max(offset_dict, key=offset_dict.get)
        hits = offset_dict[best_diff]
        
        if hits > max_hits:
            max_hits = hits
            top_track = title
            best_hist = offset_dict
            
    if max_hits < 5:
        return "No confident match found.", None
        
    return top_track, best_hist
