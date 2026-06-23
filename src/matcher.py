from collections import defaultdict

def match_query(query_hashes, database):
    match_scores = defaultdict(lambda: defaultdict(int))
    
    for hash_val, query_time in query_hashes:
        if hash_val in database.hashes:
            matches = database.hashes[hash_val]
            for song_name, db_time in matches:
                offset = db_time - query_time
                match_scores[song_name][offset] += 1
                
    best_song = None
    highest_match_count = 0
    optimal_histogram = None
    
    for song_name, offset_counts in match_scores.items():
        if not offset_counts:
            continue
            
        optimal_offset = max(offset_counts, key=offset_counts.get)
        match_count = offset_counts[optimal_offset]
        
        if match_count > highest_match_count:
            highest_match_count = match_count
            best_song = song_name
            optimal_histogram = offset_counts
            
    if highest_match_count < 5:
        return "No confident match found.", None
        
    return best_song, optimal_histogram
